#![allow(non_snake_case)]
use crate::errors::{ProofError, ProofResult};
use crate::transcript::TranscriptProtocol;
use core::iter::Iterator;
use core::ops::Mul;
use core::slice;
use curve25519_dalek::constants;
use curve25519_dalek::ristretto::{CompressedRistretto, RistrettoBasepointTable, RistrettoPoint};
use curve25519_dalek::scalar::Scalar;
use curve25519_dalek::traits::{IsIdentity, MultiscalarMul};
use merlin::Transcript;
use polynomials::Polynomial;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha3::Sha3_512;
use std::fmt;

const N: usize = 2;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(feature = "std")]
use std::vec::Vec;

#[derive(Clone)]
pub struct RistrettoBasepointTableWrapper(RistrettoBasepointTable);

impl fmt::Debug for RistrettoBasepointTableWrapper {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("RistrettoBasepointTableWrapper").finish()
    }
}

/// A collection of generator points that can be used to compute various proofs
/// in this module. To create an instance of [`ProofGens`] it is recommended to
/// call ProofGens::new(`n`), where `n` is the number of bits to be used in
/// proofs and verifications.
#[derive(Debug, Clone)]
pub struct ProofGens {
    pub n_bits: usize,
    G: RistrettoBasepointTableWrapper,
    H: Vec<RistrettoPoint>,
}

/// A bit commitment proof. This is used as part of a [`OneOfManyProof`] and
/// not meant for use on its own. A zero knowledge proof that the prover knows
/// the openings of commitments to a sequence of bits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitProof {
    pub A: CompressedRistretto,
    pub C: CompressedRistretto,
    pub D: CompressedRistretto,
    pub f_j_1: Vec<Vec<Scalar>>,
    pub z_A: Scalar,
    pub z_C: Scalar,
}

/// A zero knowledge proof of membership in a set. A prover can convince a
/// verifier that he knows the index of a commitment within a set of
/// commitments, and the opening of that commitment,
/// without revealing any information about the commitment or its location
/// within the set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OneOfManyProof {
    pub B: CompressedRistretto,
    pub bit_proof: BitProof,
    pub G_k: Polynomial<CompressedRistretto>,
    pub z: Scalar,
}

impl ProofGens {
    /// Create a new instance of [`ProofGens`] with enough generator points to
    /// support proof and verification over an `n_bit` sized set.
    ///
    /// ```
    /// # use one_of_many_proofs::proofs::ProofGens;
    /// // Support 10 bit membership proofs
    /// let gens = ProofGens::new(10);
    /// ```
    pub fn new(n_bits: usize) -> ProofResult<ProofGens> {
        if n_bits <= 1 {
            return Err(ProofError::SetIsTooSmall);
        }
        if n_bits > 32 {
            return Err(ProofError::SetIsTooLarge);
        };

        // Compute enough generator points to support vector commitments of
        // length 2*n: r*G + v[0]*H[0] + ... + v[2n-1]*H[2n-1]
        //
        // G       = Ristretto Base Point
        // H[0]    = hash(G)
        // H[1]    = hash(H[0])
        //  .           .
        //  .           .
        //  .           .
        // H[2n-1] = hash(H[2n-2])
        let mut gens = ProofGens {
            n_bits,
            G: RistrettoBasepointTableWrapper(constants::RISTRETTO_BASEPOINT_TABLE),
            H: Vec::with_capacity(N * n_bits),
        };
        gens.H.push(RistrettoPoint::hash_from_bytes::<Sha3_512>(
            gens.G.0.basepoint().compress().as_bytes(),
        ));
        for i in 1..(N * n_bits) {
            gens.H.push(RistrettoPoint::hash_from_bytes::<Sha3_512>(
                gens.H[i - 1].compress().as_bytes(),
            ));
        }
        Ok(gens)
    }

    /// Returns the maximum set size that can be processed in a proof or
    /// verification. For example, a 10 bit proof would only be able to support
    /// proofs over a set with at most `2^10 = 1024` members. Note, proofs over
    /// smaller sets will be extended by repeating the first member.
    pub fn max_set_size(&self) -> usize {
        N.checked_pow(self.n_bits as u32).unwrap()
    }

    /// Create a pedersen commitment, with value `v` and blinding factor `r`.
    pub fn commit(&self, v: &Scalar, r: &Scalar) -> ProofResult<RistrettoPoint> {
        Ok(v * self.H[0] + &self.G.0 * r)
    }

    /// Commit to the bits in `l`, and generate the corresponding proof.
    /// Note, `l` must be within the supported set size, eg, for an `n` bit
    /// proof, `l` mus reside within the range: 0 <= `l` < 2^`n`.
    ///
    /// This proof uses a [`merlin`] transcript to generate a challenge
    /// scalar for use as a non-interactive proof protocol.
    ///
    /// This function returns the bit commitment, `B`, its assosciated
    /// [`BitProof`], and the challenge scalar `x`.
    ///
    /// ```
    /// # use rand::rngs::OsRng; // You should use a more secure RNG
    /// # use one_of_many_proofs::proofs::ProofGens;
    /// # use curve25519_dalek::scalar::Scalar;
    /// # use merlin::Transcript;
    /// // Compute the generators necessary for 5 bit proofs
    /// let gens = ProofGens::new(5).unwrap();
    /// let l = 7; // Some index within the range 0 <= `l` <= 2^5
    ///
    /// // The proof requires us to provide random noise values. For secure
    /// // applications, be sure to use a more secure RNG.
    /// # let a_j = (0..1)
    /// # .map(|_| {
    /// #     (0..gens.n_bits)
    /// #         .map(|_| Scalar::random(&mut OsRng))
    /// #         .collect()
    /// # })
    /// # .collect::<Vec<Vec<Scalar>>>();
    ///
    /// // Create a new transcript and compute the bit commitment and its proof
    /// let mut t = Transcript::new(b"doctest example");
    /// let (B, proof, x) = gens.commit_bits(&mut t, l, &a_j).unwrap();
    /// ```
    pub fn commit_bits(
        &self,
        transcript: &mut Transcript,
        l: usize,
        a_j_1: &Vec<Vec<Scalar>>,
    ) -> ProofResult<(CompressedRistretto, BitProof, Scalar)> {
        if l >= self.max_set_size() {
            return Err(ProofError::IndexOutOfBounds);
        }

        transcript.bit_proof_domain_sep(self.n_bits as u64);

        // Create a `TranscriptRng` from the high-level witness data
        //
        // The prover wants to rekey the RNG with its witness data (`l`).
        let mut rng = {
            let mut builder = transcript.build_rng();

            // Commit to witness data
            builder = builder.rekey_with_witness_bytes(b"l", Scalar::from(l as u64).as_bytes());

            use rand::thread_rng;
            builder.finalize(&mut thread_rng())
        };

        let b_j_i = (0..N)
            .map(|i| {
                (0..self.n_bits)
                    .map(|j| Scalar::from(delta(bit(l, j, N), i) as u32))
                    .collect()
            })
            .collect::<Vec<Vec<Scalar>>>();

        let r_A = Scalar::random(&mut rng);
        let r_B = Scalar::random(&mut rng);
        let r_C = Scalar::random(&mut rng);
        let r_D = Scalar::random(&mut rng);

        let mut a_j_i = vec![vec![Scalar::default(); self.n_bits]; N];
        for j in 0..self.n_bits {
            a_j_i[0][j] = -a_j_1.iter().map(|a| a[j]).sum::<Scalar>();
            for i in 0..a_j_1.len() {
                a_j_i[i + 1][j] = a_j_1[i][j];
            }
        }
        let A = a_j_i.iter().flatten().commit(&self, &r_A)?.compress();
        let B = b_j_i.iter().flatten().commit(&self, &r_B)?.compress();
        let C = a_j_i
            .iter()
            .flatten()
            .zip(b_j_i.iter().flatten())
            .map(|(a, b)| a * (Scalar::one() - Scalar::from(2u32) * b))
            .commit(&self, &r_C)?
            .compress();
        let D = a_j_i
            .iter()
            .flatten()
            .map(|a| -a * a)
            .commit(&self, &r_D)?
            .compress();

        transcript.validate_and_append_point(b"A", &A)?;
        transcript.validate_and_append_point(b"B", &B)?;
        transcript.validate_and_append_point(b"C", &C)?;
        transcript.validate_and_append_point(b"D", &D)?;

        let x = transcript.challenge_scalar(b"bit-proof-challenge");

        let mut f_j_1 = vec![vec![Scalar::default(); b_j_i[0].len()]; b_j_i.len() - 1];
        for i in 1..b_j_i.len() {
            for j in 0..b_j_i[0].len() {
                f_j_1[i - 1][j] = b_j_i[i][j] * x + a_j_i[i][j];
            }
        }

        let z_A = r_B * x + r_A;
        let z_C = r_C * x + r_D;

        for f in f_j_1.iter().flatten() {
            transcript.append_scalar(b"f1_j", f);
        }
        transcript.append_scalar(b"z_A", &z_A);
        transcript.append_scalar(b"z_C", &z_C);

        Ok((
            B,
            BitProof {
                A,
                C,
                D,
                f_j_1,
                z_A,
                z_C,
            },
            x,
        ))
    }

    /// Verify a bit commitment proof.
    ///
    /// ```
    /// # use rand::rngs::OsRng; // You should use a more secure RNG
    /// # use one_of_many_proofs::proofs::ProofGens;
    /// # use curve25519_dalek::scalar::Scalar;
    /// # use merlin::Transcript;
    /// # let gens = ProofGens::new(5).unwrap();
    /// # let l = 7; // Some index within the range 0 <= `l` <= 2^5
    /// # let a_j = (0..1)
    /// # .map(|_| {
    /// #   (0..gens.n_bits)
    /// #        .map(|_| Scalar::random(&mut OsRng))
    /// #        .collect()
    /// # })
    /// # .collect::<Vec<Vec<Scalar>>>();
    /// # let mut t = Transcript::new(b"doctest example");
    /// # let (B, proof, _) = gens.commit_bits(&mut t, l, &a_j).unwrap();
    /// // Create new transcript and verify a bit commitment against its proof
    /// let mut t = Transcript::new(b"doctest example");
    /// assert!(gens.verify_bits(&mut t, &B, &proof).is_ok());
    /// ```
    pub fn verify_bits(
        &self,
        transcript: &mut Transcript,
        B: &CompressedRistretto,
        proof: &BitProof,
    ) -> ProofResult<Scalar> {
        transcript.bit_proof_domain_sep(self.n_bits as u64);

        transcript.validate_and_append_point(b"A", &proof.A)?;
        transcript.validate_and_append_point(b"B", &B)?;
        transcript.validate_and_append_point(b"C", &proof.C)?;
        transcript.validate_and_append_point(b"D", &proof.D)?;

        let x = transcript.challenge_scalar(b"bit-proof-challenge");

        for f in proof.f_j_1.iter().flatten() {
            transcript.append_scalar(b"f1_j", f);
        }
        transcript.append_scalar(b"z_A", &proof.z_A);
        transcript.append_scalar(b"z_C", &proof.z_C);

        // Verify proof size
        if proof.f_j_1[0].len() != self.n_bits {
            return Err(ProofError::InvalidProofSize);
        }

        // Verify all scalars are canonical
        for f in proof.f_j_1.iter().flatten() {
            if !f.is_canonical() {
                return Err(ProofError::InvalidScalar(*f));
            }
        }
        if !proof.z_A.is_canonical() {
            return Err(ProofError::InvalidScalar(proof.z_A));
        }
        if !proof.z_C.is_canonical() {
            return Err(ProofError::InvalidScalar(proof.z_C));
        }

        // Inflate f1_j to include reconstructed f_j vector
        let f_j_1 = &proof.f_j_1;
        let mut f_j_i = vec![vec![Scalar::default(); self.n_bits]; N];
        for j in 0..self.n_bits {
            f_j_i[0][j] = x - f_j_1.iter().map(|f| f[j]).sum::<Scalar>();
            for i in 0..f_j_1.len() {
                f_j_i[i + 1][j] = f_j_1[i][j];
            }
        }

        // Verify relation R1
        if x * B.decompress().unwrap() + proof.A.decompress().unwrap()
            != f_j_i.iter().flatten().commit(&self, &proof.z_A)?
        {
            return Err(ProofError::VerificationFailed);
        }
        let r1 = f_j_i
            .iter()
            .map(|f_j| f_j.iter().map(|f| f * (x - f)))
            .flatten()
            .commit(&self, &proof.z_C)?;
        if x * proof.C.decompress().unwrap() + proof.D.decompress().unwrap() != r1 {
            return Err(ProofError::VerificationFailed);
        }
        Ok(x)
    }
}

pub trait OneOfManyProofs {
    //! Trait for computing and verifying OneOfMany zero-knowledge membership
    //! proofs over a set of points. Each method is designed to iterate over a
    //! set of [`RistrettoPoint`]s representing pedersen commitments. A prover
    //! should know the opening of one commitment in the set, and the index of
    //! that commitment within the set.
    //!
    //! # Proof of knowledge of a commitment that opens to zero
    //! The `prove()` and `verify()` methods may be used to compute and verify
    //! membership of a commitment that opens to zero within a specified set of
    //! commitments. Proofs for commitments are demonstrated further below.
    //!
    //! ```
    //! # use rand::rngs::OsRng; // You should use a more secure RNG
    //! # use one_of_many_proofs::proofs::{ProofGens, OneOfManyProofs};
    //! # use curve25519_dalek::scalar::Scalar;
    //! # use curve25519_dalek::ristretto::RistrettoPoint;
    //! # use merlin::Transcript;
    //! #
    //! // Set up proof generators
    //! let gens = ProofGens::new(5).unwrap();
    //!
    //! // Create the prover's commitment to zero
    //! let l: usize = 3; // The prover's commitment will be third in the set
    //! let v = Scalar::zero();
    //! let r = Scalar::random(&mut OsRng); // You should use a more secure RNG
    //! let C_l = gens.commit(&v, &r).unwrap();
    //!
    //! // Build a random set containing the prover's commitment at index `l`
    //! let mut set = (1..gens.max_set_size())
    //!     .map(|_| RistrettoPoint::random(&mut OsRng))
    //!     .collect::<Vec<RistrettoPoint>>();
    //! set.insert(l, C_l);
    //!
    //! // Compute a `OneOfMany` membership proof for this commitment
    //! let mut t = Transcript::new(b"OneOfMany-Test");
    //! let proof = set.prove(&gens, &mut t.clone(), l, &r).unwrap();
    //!
    //! // Verify this membership proof, without any knowledge of `l` or `r`.
    //! assert!(set
    //!     .verify(&gens, &mut t.clone(), &proof)
    //!     .is_ok());
    //! ```
    //!
    //! # Proof of knowledge of a commitment that opens to any value
    //! This is an extension of the OneOfMany proof protocol, to enable proof
    //! of knowledge of a commitment to any value within the set.
    //!
    //! If the prover knows the index, `l`, and opening of some commitment,
    //! `C_l`, within a set of commitments, then the prover can prove this
    //! knowledge, and also prove that some new commitment, `C_new`, commits to
    //! the same value as `C_l`.
    //!
    //! The basic concept is that `C_new` will be supplied as an offset in the
    //! proof computation. This offset will be subtracted from every member in
    //! the set before computing a OneOfMany proof. Since, `C_new` commits to
    //! the same value as `C_l`, the result after subtraction will yield a
    //! commitment to 0 at index `l` within the set. This commitment to 0 can
    //! now be used to compute a OneOfMany proof of membership.
    //!
    //! > Note: the resultant commitment to zero is blinded by `r - r_new`, and
    //! this value must be supplied instead of `r`, to compute the proof.
    //!
    //! ```
    //! # use rand::rngs::OsRng; // You should use a more secure RNG
    //! # use one_of_many_proofs::proofs::{ProofGens, OneOfManyProofs};
    //! # use curve25519_dalek::scalar::Scalar;
    //! # use curve25519_dalek::ristretto::RistrettoPoint;
    //! # use merlin::Transcript;
    //! #
    //! // Set up proof generators
    //! let gens = ProofGens::new(5).unwrap();
    //!
    //! // Create the prover's commitment to a random value
    //! let l: usize = 3; // The prover's commitment will be third in the set
    //! let v = Scalar::random(&mut OsRng); // You should use a more secure RNG
    //! let r = Scalar::random(&mut OsRng); // You should use a more secure RNG
    //! let C_l = gens.commit(&v, &r).unwrap();
    //!
    //! // Build a random set containing the prover's commitment at index `l`
    //! let mut set = (1..gens.max_set_size())
    //!     .map(|_| RistrettoPoint::random(&mut OsRng))
    //!     .collect::<Vec<RistrettoPoint>>();
    //! set.insert(l, C_l);
    //!
    //! // Create a new commitment to the same value as `C_l`
    //! let r_new = Scalar::random(&mut OsRng); // You should use a more secure RNG
    //! let C_new = gens.commit(&v, &r_new).unwrap();
    //!
    //! // Compute a `OneOfMany` membership proof for this commitment
    //! let mut t = Transcript::new(b"OneOfMany-Test");
    //! let proof = set.prove_with_offset(&gens, &mut t.clone(), l, &(r - r_new), Some(&C_new)).unwrap();
    //!
    //! // Verify this membership proof, without any knowledge of `l` or `r`.
    //! assert!(set
    //!     .verify_with_offset(&gens, &mut t.clone(), &proof, Some(&C_new))
    //!     .is_ok());
    //! ```

    /// Prove knowledge of a commitment opening to zero. The prover must
    /// provide the index `l` of a commitment within the set that opens to
    /// zero, and  also its blinding factor, `r`.
    ///
    /// Note: this is just a convenience wrapper around `prove_with_offset()`.
    fn prove(
        &self,
        gens: &ProofGens,
        transcript: &mut Transcript,
        l: usize,
        r: &Scalar,
    ) -> ProofResult<OneOfManyProof> {
        self.prove_with_offset(gens, transcript, l, r, None)
    }

    /// Verify a proof of knowledge of a commitment opening to zero. This
    /// verification will only succeed if the proven commitment opens to zero.
    ///
    /// Note: this is just a convenience wrapper around `verify_with_offset()`.
    fn verify(
        &self,
        gens: &ProofGens,
        transcript: &mut Transcript,
        proof: &OneOfManyProof,
    ) -> ProofResult<()> {
        self.verify_with_offset(gens, transcript, proof, None)
    }

    /// Prove knowledge of a commitment opening to any value.
    fn prove_with_offset(
        &self,
        gens: &ProofGens,
        transcript: &mut Transcript,
        l: usize,
        r: &Scalar,
        offset: Option<&RistrettoPoint>,
    ) -> ProofResult<OneOfManyProof>;

    /// Verify a proof of knowledge of a commitment opening to any value.
    fn verify_with_offset(
        &self,
        gens: &ProofGens,
        transcript: &mut Transcript,
        proof: &OneOfManyProof,
        offset: Option<&RistrettoPoint>,
    ) -> ProofResult<()> {
        self.verify_batch_with_offsets(
            gens,
            transcript,
            slice::from_ref(&proof),
            slice::from_ref(&offset),
        )
    }

    /// Batch verification of membership proofs
    fn verify_batch(
        &self,
        gens: &ProofGens,
        transcript: &mut Transcript,
        proofs: &[OneOfManyProof],
    ) -> ProofResult<()> {
        self.verify_batch_with_offsets(gens, transcript, proofs, &vec![None; proofs.len()])
    }

    /// Batch verification of membership proofs
    fn verify_batch_with_offsets(
        &self,
        gens: &ProofGens,
        transcript: &mut Transcript,
        proofs: &[OneOfManyProof],
        offsets: &[Option<&RistrettoPoint>],
    ) -> ProofResult<()>;
}

impl OneOfManyProofs for Vec<RistrettoPoint> {
    fn prove_with_offset(
        &self,
        gens: &ProofGens,
        transcript: &mut Transcript,
        l: usize,
        r: &Scalar,
        offset: Option<&RistrettoPoint>,
    ) -> ProofResult<OneOfManyProof> {
        transcript.one_of_many_proof_domain_sep(gens.n_bits as u64);

        // Create a `TranscriptRng` from the high-level witness data
        //
        // The prover wants to rekey the RNG with its witness data (`l` and `r`).
        let mut rng = {
            let mut builder = transcript.build_rng();

            // Commit to witness data
            builder = builder.rekey_with_witness_bytes(b"l", Scalar::from(l as u64).as_bytes());
            builder = builder.rekey_with_witness_bytes(b"r", r.as_bytes());

            use rand::thread_rng;
            builder.finalize(&mut thread_rng())
        };

        if l > gens.max_set_size() {
            return Err(ProofError::IndexOutOfBounds);
        }

        let rho_k = (0..gens.n_bits)
            .map(|_| Scalar::random(&mut rng))
            .collect::<Vec<Scalar>>();
        let a_j_1 = (0..N - 1)
            .map(|_| (0..gens.n_bits).map(|_| Scalar::random(&mut rng)).collect())
            .collect::<Vec<Vec<Scalar>>>();

        let mut a_j_i = vec![vec![Scalar::default(); gens.n_bits]; N];
        for j in 0..gens.n_bits {
            a_j_i[0][j] = -a_j_1.iter().map(|a| a[j]).sum::<Scalar>();
            for i in 0..a_j_1.len() {
                a_j_i[i + 1][j] = a_j_1[i][j];
            }
        }

        let G_k = Polynomial::from(
            rho_k
                .iter()
                .map(|rho| gens.commit(&Scalar::zero(), rho).unwrap())
                .collect::<Vec<RistrettoPoint>>(),
        );

        let points_minus_offset = self
            .iter()
            .map(|&C_i| if let Some(O) = offset { C_i - O } else { C_i })
            .collect::<Vec<RistrettoPoint>>();

        let p_i_k = (0..self.len())
            .map(|i| compute_p_i(i, l, &a_j_i))
            .collect::<Vec<Vec<Scalar>>>();
        if p_i_k.len() < gens.max_set_size() {
            return Err(ProofError::SetIsTooSmall);
        } else if p_i_k.len() > gens.max_set_size() {
            return Err(ProofError::SetIsTooLarge);
        }
        let p_k_i = transpose(&p_i_k);

        // I need the underlying vector but the library doesn't provide a method for that....
        let mut G_k = G_k.iter().map(|el| *el).collect::<Vec<RistrettoPoint>>();
        G_k.par_iter_mut().enumerate().for_each(|(k, coeff)| {
            *coeff += RistrettoPoint::multiscalar_mul(&p_k_i[k], &points_minus_offset);
        });
        // Compress the points to save space
        let compressed_G_k = G_k
            .iter()
            .map(|el| el.compress())
            .collect::<Vec<CompressedRistretto>>();
        // Back to a polynomial...
        let G_k = Polynomial::from(compressed_G_k);

        for k in 0..gens.n_bits - 1 {
            transcript.validate_and_append_point(b"G_k", &G_k[k])?;
        }

        let (B, bit_proof, x) = gens.commit_bits(&mut transcript.clone(), l, &a_j_1)?;

        let z = r * scalar_exp(x, gens.n_bits) - Polynomial::from(rho_k).eval(x).unwrap();

        transcript.append_scalar(b"z", &z);

        Ok(OneOfManyProof {
            B,
            bit_proof,
            G_k,
            z,
        })
    }

    fn verify_batch_with_offsets(
        &self,
        gens: &ProofGens,
        transcript: &mut Transcript,
        proofs: &[OneOfManyProof],
        offsets: &[Option<&RistrettoPoint>],
    ) -> ProofResult<()> {
        transcript.one_of_many_proof_domain_sep(gens.n_bits as u64);

        // Every proof must have an entry in `offsets`, even if it is `None`.
        if proofs.len() != offsets.len() {
            return Err(ProofError::VerificationFailed);
        }

        let x_vec: Vec<_> = proofs
            .iter()
            .map(|p| {
                if !p.z.is_canonical() {
                    return Err(ProofError::InvalidScalar(p.z));
                }

                let mut t = transcript.clone();
                for k in 0..gens.n_bits - 1 {
                    t.validate_and_append_point(b"G_k", &p.G_k[k])?;
                }
                gens.verify_bits(&mut t, &p.B, &p.bit_proof)
            })
            .collect::<Result<Vec<_>, _>>()?;

        // create 3d array: [proof, n, m] <- this is a cry for help
        let mut f_p_j_i = vec![
            vec![vec![Scalar::default(); proofs[0].bit_proof.f_j_1[0].len()]; N];
            proofs.len()
        ];
        for k in 0..proofs.len() {
            // Inflate f1_j to include reconstructed f_j vector
            let f_j_1 = &proofs[k].bit_proof.f_j_1;
            let mut f_j_i = vec![vec![Scalar::default(); f_j_1[0].len()]; N];
            for j in 0..f_j_1[0].len() {
                f_j_i[0][j] = x_vec[k] - f_j_1.iter().map(|f| f[j]).sum::<Scalar>();
                for i in 0..f_j_1.len() {
                    f_j_i[i + 1][j] = f_j_1[i][j];
                }
            }
            f_p_j_i[k] = f_j_i;
        }

        // Using batch verification strategy from: https://eprint.iacr.org/2019/373.pdf
        let coeffs = self
            .par_iter()
            .enumerate()
            .map(|(i, _)| {
                proofs
                    .iter()
                    .enumerate()
                    .map(|(p, _)| {
                        (0..gens.n_bits)
                            .map(|j| f_p_j_i[p][bit(i, j, N)][j])
                            .product::<Scalar>()
                    })
                    .collect::<Vec<Scalar>>()
            })
            .collect::<Vec<Vec<Scalar>>>();

        let horizontal_sum = coeffs
            .par_iter()
            .map(|vec| vec.iter().sum::<Scalar>())
            .collect::<Vec<Scalar>>();

        let vertical_sum = (0..coeffs[0].len())
            .into_iter()
            .map(|i| {
                coeffs
                    .par_iter()
                    .map(|row| row[i])
                    .reduce(|| Scalar::zero(), |a, b| a + b)
            })
            .collect::<Vec<Scalar>>();

        let O = offsets
            .iter()
            .filter_map(|O| if let Some(O) = O { Some(O) } else { None })
            .map(|&O| O)
            .collect::<Vec<_>>()
            .iter()
            .enumerate()
            .map(|(k, &O)| O * vertical_sum[k])
            .sum::<RistrettoPoint>();

        if self.len() < gens.max_set_size() {
            return Err(ProofError::SetIsTooSmall);
        } else if self.len() > gens.max_set_size() {
            return Err(ProofError::SetIsTooLarge);
        }

        // decompress points in G_k polnyomial... ugly sight...
        let G_k_vec = proofs
            .iter()
            .map(|p| {
                p.G_k
                    .iter()
                    .map(|point| point.decompress().unwrap())
                    .collect::<Vec<RistrettoPoint>>()
            })
            .collect::<Vec<Vec<RistrettoPoint>>>();

        let C = RistrettoPoint::multiscalar_mul(horizontal_sum, self);
        let E = gens.commit(&Scalar::zero(), &proofs.iter().map(|p| p.z).sum())?;
        let G = G_k_vec
            .iter()
            .zip(x_vec.iter())
            .map(|(G_k, &x)| Polynomial::from(G_k.clone()).eval(x).unwrap())
            .sum::<RistrettoPoint>();
        if C.is_identity() || E.is_identity() || G.is_identity() {
            return Err(ProofError::VerificationFailed);
        }
        if C != E + G + O {
            return Err(ProofError::VerificationFailed);
        }
        Ok(())
    }
}

trait VectorCommit {
    fn commit(self, gens: &ProofGens, r: &Scalar) -> ProofResult<RistrettoPoint>;
}

impl<I, T> VectorCommit for I
where
    I: Iterator<Item = T>,
    T: core::borrow::Borrow<curve25519_dalek::scalar::Scalar>
        + Mul<RistrettoPoint, Output = RistrettoPoint>,
{
    fn commit(self, gens: &ProofGens, r: &Scalar) -> ProofResult<RistrettoPoint> {
        let c = r * &gens.G.0;
        let mut scalars = Vec::with_capacity(gens.H.len());
        for (i, v) in self.enumerate() {
            if i >= gens.H.len() {
                return Err(ProofError::SetIsTooLarge);
            }
            scalars.push(v);
        }
        let c_i = RistrettoPoint::multiscalar_mul(scalars, gens.H.as_slice());
        Ok(c + c_i)
    }
}

fn compute_p_i(i: usize, l: usize, a_j_i: &Vec<Vec<Scalar>>) -> Vec<Scalar> {
    assert!(a_j_i.len() == N); // Must have two rows of random scalars
    assert!(a_j_i[0].len() == a_j_i[1].len()); // Make sure each row is the same length
    let n_bits = a_j_i[0].len();

    // Create polynomial vector
    let mut p = Polynomial::from(Vec::with_capacity(n_bits));
    p.push(Scalar::one());

    // Multiply each polynomial
    for j in 0..n_bits {
        let mut f = Polynomial::new();
        f.push(a_j_i[bit(i, j, N)][j]);
        if 0 != delta(bit(l, j, N), bit(i, j, N)) {
            f.push(Scalar::one());
        }
        p *= f;
    }

    // Resize the vector to be M bits wide
    let mut v: Vec<Scalar> = p.into();
    v.resize_with(n_bits, || Scalar::zero());
    v
}

fn transpose(matrix: &Vec<Vec<Scalar>>) -> Vec<Vec<Scalar>> {
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut transposed_matrix = vec![vec![Scalar::zero(); rows]; cols];

    for i in 0..rows {
        for j in 0..cols {
            transposed_matrix[j][i] = matrix[i][j];
        }
    }
    transposed_matrix
}

fn scalar_exp(base: Scalar, exp: usize) -> Scalar {
    let mut res = Scalar::one();
    for _ in 0..exp {
        res *= base;
    }
    res
}

fn bit(v: usize, j: usize, n: usize) -> usize {
    (v / n.pow(j as u32)) % n
}

fn delta(a: usize, b: usize) -> usize {
    if a == b {
        1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use crate::errors::*;
    use crate::proofs::*;
    use curve25519_dalek::ristretto::RistrettoPoint;
    use curve25519_dalek::scalar::Scalar;
    use merlin::Transcript;
    use rand::rngs::OsRng; // You should use a more secure RNG

    #[test]
    fn new_generators() {
        assert!(ProofGens::new(5).is_ok());
        assert_eq!(ProofGens::new(0).unwrap_err(), ProofError::SetIsTooSmall);
        assert_eq!(ProofGens::new(1).unwrap_err(), ProofError::SetIsTooSmall);
        assert!(ProofGens::new(32).is_ok());
        assert_eq!(ProofGens::new(33).unwrap_err(), ProofError::SetIsTooLarge);
        assert_eq!(
            ProofGens::new(0xffffffff).unwrap_err(),
            ProofError::SetIsTooLarge
        );
    }

    #[test]
    fn bit_commitments() {
        // Set up proof generators
        let gens = ProofGens::new(5).unwrap();

        // Create the prover's commitment to zero
        let l: usize = 3; // The prover's commitment will be third in the set
        let t = Transcript::new(b"OneOfMany-Test");

        let a_j_1 = (0..N - 1)
            .map(|_| {
                (0..gens.n_bits)
                    .map(|_| Scalar::random(&mut OsRng))
                    .collect()
            })
            .collect::<Vec<Vec<Scalar>>>();

        // Compute a bit commitment and its proof
        let (B, proof, _) = gens.commit_bits(&mut t.clone(), l, &a_j_1).unwrap();
        assert!(gens.verify_bits(&mut t.clone(), &B, &proof).is_ok());

        // Check error if index out of bounds
        assert_eq!(
            gens.commit_bits(&mut t.clone(), gens.max_set_size(), &a_j_1)
                .unwrap_err(),
            ProofError::IndexOutOfBounds
        );
    }

    #[test]
    fn prove_single() {
        // Set up proof generators
        let gens = ProofGens::new(5).unwrap();

        // Create the prover's commitment to zero
        let l: usize = 3; // The prover's commitment will be third in the set
        let v = Scalar::zero();
        let r = Scalar::random(&mut OsRng); // You should use a more secure RNG
        let C_l = gens.commit(&v, &r).unwrap();

        // Build a random set containing the prover's commitment at index `l`
        let mut set = (1..gens.max_set_size())
            .map(|_| RistrettoPoint::random(&mut OsRng))
            .collect::<Vec<RistrettoPoint>>();
        set.insert(l, C_l);
        let t = Transcript::new(b"OneOfMany-Test");

        // Compute a `OneOfMany` membership proof for this commitment
        let proof = set.prove(&gens, &mut t.clone(), l, &r).unwrap();
        assert!(set.verify(&gens, &mut t.clone(), &proof).is_ok());

        // Check error if index out of bounds
        assert_eq!(
            set.prove(&gens, &mut t.clone(), gens.max_set_size(), &r)
                .unwrap_err(),
            ProofError::IndexOutOfBounds
        );

        // Prove should fail if set too small or too large
        let removed = set.pop().unwrap();
        assert_eq!(
            set.prove(&gens, &mut t.clone(), l, &r).unwrap_err(),
            ProofError::SetIsTooSmall
        );
        set.push(RistrettoPoint::random(&mut OsRng));
        assert!(set.prove(&gens, &mut t.clone(), l, &r).is_ok()); // Ok!
        set.push(RistrettoPoint::random(&mut OsRng));
        assert_eq!(
            set.prove(&gens, &mut t.clone(), l, &r).unwrap_err(),
            ProofError::SetIsTooLarge
        );

        // Return set to original state
        set.pop();
        set.pop();
        set.push(removed);

        // Verify should fail if set has been modified
        let removed = set.pop().unwrap();
        assert_eq!(
            set.verify(&gens, &mut t.clone(), &proof).unwrap_err(),
            ProofError::SetIsTooSmall
        );
        set.push(RistrettoPoint::random(&mut OsRng));
        assert_eq!(
            set.verify(&gens, &mut t.clone(), &proof).unwrap_err(),
            ProofError::VerificationFailed
        );
        set.push(RistrettoPoint::random(&mut OsRng));
        assert_eq!(
            set.verify(&gens, &mut t.clone(), &proof).unwrap_err(),
            ProofError::SetIsTooLarge
        );

        // Return set to original state
        set.pop();
        set.pop();
        set.push(removed);
    }

    #[test]
    fn prove_single_with_offset() {
        // Set up proof generators
        let gens = ProofGens::new(5).unwrap();

        // Create the prover's commitment to zero
        let l: usize = 3; // The prover's commitment will be third in the set
        let v = Scalar::zero();
        let r = Scalar::random(&mut OsRng); // You should use a more secure RNG
        let C_l = gens.commit(&v, &r).unwrap();

        // Build a random set containing the prover's commitment at index `l`
        let mut set = (1..gens.max_set_size())
            .map(|_| RistrettoPoint::random(&mut OsRng))
            .collect::<Vec<RistrettoPoint>>();
        set.insert(l, C_l);

        let t = Transcript::new(b"OneOfMany-Test");

        // First test with no offest
        let proof = set
            .prove_with_offset(&gens, &mut t.clone(), l, &r, None)
            .unwrap();
        assert!(set
            .verify_with_offset(&gens, &mut t.clone(), &proof, None)
            .is_ok());

        // Now replace C_l with a committment to a non-zero value
        let v = Scalar::random(&mut OsRng); // You should use a more secure RNG
        let C_l = gens.commit(&v, &r).unwrap();
        set[l] = C_l;

        // Compute new commitment, to same value as `C_l`
        let r_new = Scalar::random(&mut OsRng);
        let C_new = gens.commit(&v, &r_new).unwrap(); // New commitment to same value

        // Now test with the valid offset and commitment to non-zero
        let proof = set
            .prove_with_offset(&gens, &mut t.clone(), l, &(r - r_new), Some(&C_new))
            .unwrap();
        assert!(set
            .verify_with_offset(&gens, &mut t.clone(), &proof, Some(&C_new))
            .is_ok());

        // Now test with the incorrect offset
        assert_eq!(
            set.verify_with_offset(
                &gens,
                &mut t.clone(),
                &proof,
                Some(&RistrettoPoint::random(&mut OsRng))
            )
            .unwrap_err(),
            ProofError::VerificationFailed
        );

        // Check error if index out of bounds
        assert_eq!(
            set.prove_with_offset(&gens, &mut t.clone(), gens.max_set_size(), &r, Some(&C_new))
                .unwrap_err(),
            ProofError::IndexOutOfBounds
        );

        // Prove should fail if set too small or too large
        let removed = set.pop().unwrap();
        assert_eq!(
            set.prove_with_offset(&gens, &mut t.clone(), l, &r, Some(&C_new))
                .unwrap_err(),
            ProofError::SetIsTooSmall
        );
        set.push(RistrettoPoint::random(&mut OsRng));
        assert!(set
            .prove_with_offset(&gens, &mut t.clone(), l, &r, Some(&C_new))
            .is_ok()); // Ok!
        set.push(RistrettoPoint::random(&mut OsRng));
        assert_eq!(
            set.prove_with_offset(&gens, &mut t.clone(), l, &r, Some(&C_new))
                .unwrap_err(),
            ProofError::SetIsTooLarge
        );

        // Return set to original state
        set.pop();
        set.pop();
        set.push(removed);

        // Verify should fail if set has been modified
        let removed = set.pop().unwrap();
        assert_eq!(
            set.verify_with_offset(&gens, &mut t.clone(), &proof, Some(&C_new))
                .unwrap_err(),
            ProofError::SetIsTooSmall
        );
        set.push(RistrettoPoint::random(&mut OsRng));
        assert_eq!(
            set.verify_with_offset(&gens, &mut t.clone(), &proof, Some(&C_new))
                .unwrap_err(),
            ProofError::VerificationFailed
        );
        set.push(RistrettoPoint::random(&mut OsRng));
        assert_eq!(
            set.verify_with_offset(&gens, &mut t.clone(), &proof, Some(&C_new))
                .unwrap_err(),
            ProofError::SetIsTooLarge
        );

        // Return set to original state
        set.pop();
        set.pop();
        set.push(removed);
    }

    #[test]
    fn prove_batch() {
        // Set up proof generators
        let gens = ProofGens::new(5).unwrap();

        // Create the prover's commitment to zero
        let l: usize = 3; // The prover's commitment will be third in the set
        let v = Scalar::random(&mut OsRng); // You should use a more secure RNG
        let r = Scalar::random(&mut OsRng); // You should use a more secure RNG
        let C_l = gens.commit(&v, &r).unwrap();

        // Compute new commitment, to same value as `C_l`
        let r_new = Scalar::random(&mut OsRng);
        let C_new = gens.commit(&v, &r_new).unwrap(); // New commitment to same value

        // Build a random set containing the prover's commitment at index `l`
        let mut set = (1..gens.max_set_size())
            .map(|_| RistrettoPoint::random(&mut OsRng))
            .collect::<Vec<RistrettoPoint>>();
        set.insert(l, C_l);

        let t = Transcript::new(b"OneOfMany-Test");

        // Verify batch with offsets
        let mut proofs = Vec::new();
        let mut offsets = Vec::new();
        for _ in 0..10 {
            proofs.push(
                set.prove_with_offset(&gens, &mut t.clone(), l, &(r - r_new), Some(&C_new))
                    .unwrap(),
            );
            offsets.push(Some(&C_new));
        }
        assert!(set
            .verify_batch_with_offsets(&gens, &mut t.clone(), &proofs, &offsets)
            .is_ok());

        // Now replace C_l with a committment to zero
        let v = Scalar::zero();
        let C_l = gens.commit(&v, &r).unwrap();
        set[l] = C_l;

        // Now verify batch without offsets
        let mut proofs = Vec::new();
        for _ in 0..10 {
            proofs.push(set.prove(&gens, &mut t.clone(), l, &r).unwrap());
        }
        assert!(set.verify_batch(&gens, &mut t.clone(), &proofs).is_ok());
    }

    #[test]
    fn serde() {
        // Set up proof generators
        let gens = ProofGens::new(5).unwrap();

        // Create the prover's commitment to zero
        let l: usize = 3; // The prover's commitment will be third in the set
        let v = Scalar::random(&mut OsRng); // You should use a more secure RNG
        let r = Scalar::random(&mut OsRng); // You should use a more secure RNG
        let C_l = gens.commit(&v, &r).unwrap();

        // Compute new commitment, to same value as `C_l`
        let r_new = Scalar::random(&mut OsRng);
        let C_new = gens.commit(&v, &r_new).unwrap(); // New commitment to same value

        // Build a random set containing the prover's commitment at index `l`
        let mut set = (1..gens.max_set_size())
            .map(|_| RistrettoPoint::random(&mut OsRng))
            .collect::<Vec<RistrettoPoint>>();
        set.insert(l, C_l);

        let t = Transcript::new(b"OneOfMany-Test");

        // Verify batch with offsets
        let mut proofs = Vec::new();
        let mut offsets = Vec::new();
        for _ in 0..10 {
            proofs.push(
                set.prove_with_offset(&gens, &mut t.clone(), l, &(r - r_new), Some(&C_new))
                    .unwrap(),
            );
            offsets.push(Some(&C_new));
        }
        let serialized = serde_cbor::to_vec(&proofs).unwrap();
        let proofs: Vec<OneOfManyProof> = serde_cbor::from_slice(&serialized[..]).unwrap();
        assert!(set
            .verify_batch_with_offsets(&gens, &mut t.clone(), &proofs, &offsets)
            .is_ok());

        // Now replace C_l with a committment to zero
        let v = Scalar::zero();
        let C_l = gens.commit(&v, &r).unwrap();
        set[l] = C_l;

        // Now verify batch without offsets
        let mut proofs = Vec::new();
        for _ in 0..10 {
            proofs.push(set.prove(&gens, &mut t.clone(), l, &r).unwrap());
        }
        let serialized = serde_cbor::to_vec(&proofs).unwrap();
        let proofs: Vec<OneOfManyProof> = serde_cbor::from_slice(&serialized[..]).unwrap();
        assert!(set.verify_batch(&gens, &mut t.clone(), &proofs).is_ok());
    }
}
