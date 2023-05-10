#![allow(non_snake_case)]
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use curve25519_dalek::ristretto::{CompressedRistretto, RistrettoPoint};
use curve25519_dalek::scalar::Scalar;
use merlin::Transcript;
use one_of_many_proofs::proofs::*;
use rand::rngs::OsRng;

fn generate_set(base: usize, pow: usize) -> (ProofGens, Scalar, Vec<RistrettoPoint>) {
    let l: usize = 1; // Index within the set, of the prover's commitment
    let v = Scalar::zero(); // Must prove commitment to zero
    let r = Scalar::random(&mut OsRng); // Blinding factor for prover's commitment

    let gens = ProofGens::new(base, pow).unwrap(); // Set generators
    let C_l = gens.commit(&v, &r).unwrap(); // Prover's commitment

    // Build a random set containing the prover's commitment at index `l`
    let mut set = (1..gens.max_set_size())
        .map(|_| RistrettoPoint::random(&mut OsRng))
        .collect::<Vec<RistrettoPoint>>();
    set.insert(l, C_l);
    (gens, r, set)
}

fn gen_proofs(
    n: usize,
    gens: &ProofGens,
    r: &Scalar,
    set: &Vec<RistrettoPoint>,
) -> Vec<OneOfManyProof> {
    let mut proofs = Vec::new();
    let mut tscpt = Transcript::new(b"doctest example");
    let proof = set.prove(&gens, &mut tscpt, 1, &r).unwrap();
    for _ in 0..n {
        proofs.push(proof.clone());
    }
    proofs
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("ooom");

    let transcript = Transcript::new(b"doctest example");
    let input_sets = [
        (2, 13),
        (4, 7),
        (8, 5),
        (4, 8),
        (10, 5),
        (8, 6),
        (64, 3),
        (10, 6),
    ]
    .iter()
    .map(|(base, pow)| generate_set(*base, *pow))
    .collect::<Vec<_>>();

    for i in &input_sets {
        group.bench_with_input(
            BenchmarkId::new(
                "Single prove",
                format!(
                    "Set size: {:?}, n: {:?}, m: {:?}",
                    i.2.len(),
                    i.0.n_base,
                    i.0.n_bits
                ),
            ),
            &i,
            |b, (gens, r, set)| {
                b.iter(|| {
                    let mut tscpt = transcript.clone();
                    set.prove(&gens, &mut tscpt, 1, &r).unwrap();
                })
            },
        );
    }

    let input_proofs = input_sets
        .iter()
        .map(|(gens, r, set)| {
            let mut tscpt = transcript.clone();
            let proof = set.prove(&gens, &mut tscpt, 1, &r).unwrap();
            println!(
                "n: {}, m: {}, size: {}",
                gens.n_base,
                gens.n_bits,
                calc_proof_size(&proof)
            );
            proof
        })
        .collect::<Vec<_>>();

    for (proof, input) in input_proofs.iter().zip(&input_sets) {
        let (gens, _, set) = input;
        group.bench_with_input(
            BenchmarkId::new(
                "Single verify",
                format!(
                    "Set size: {:?}, n: {:?}, m: {:?}",
                    input.2.len(),
                    input.0.n_base,
                    input.0.n_bits
                ),
            ),
            &(proof, gens, set),
            |b, (proof, gens, set)| {
                b.iter(|| {
                    let mut tscpt: Transcript = transcript.clone();
                    set.verify(gens, &mut tscpt, proof).unwrap()
                })
            },
        );
    }

    let (gens, r, set) = input_sets[4].clone();
    let proofs = gen_proofs(1000, &gens, &r, &set);
    let batches = [5, 10, 50, 100, 500, 1000];

    for batch in batches {
        group.bench_with_input(
            BenchmarkId::new("Batch verify", batch),
            &batch,
            |b, batch| {
                b.iter(|| {
                    let mut tscpt: Transcript = transcript.clone();
                    assert!(set
                        .verify_batch(
                            black_box(&gens),
                            black_box(&mut tscpt),
                            black_box(&proofs[..*batch]),
                        )
                        .is_ok());
                })
            },
        );
    }
}

fn calc_proof_size(proof: &OneOfManyProof) -> usize {
    let G_k = proof
        .G_k
        .iter()
        .map(|el: &CompressedRistretto| *el)
        .collect::<Vec<CompressedRistretto>>();

    let proof_size = std::mem::size_of_val(&*G_k)
        + std::mem::size_of_val(&proof.B)
        + std::mem::size_of_val(&proof.z);

    let bit_proof_size = std::mem::size_of_val(
        &*proof
            .bit_proof
            .f_j_1
            .iter()
            .flatten()
            .map(|el| *el)
            .collect::<Vec<Scalar>>(),
    ) + std::mem::size_of_val(&proof.bit_proof.A)
        + std::mem::size_of_val(&proof.bit_proof.C)
        + std::mem::size_of_val(&proof.bit_proof.D)
        + std::mem::size_of_val(&proof.bit_proof.z_A)
        + std::mem::size_of_val(&proof.bit_proof.z_C);

    proof_size + bit_proof_size
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = criterion_benchmark
}
criterion_main!(benches);
