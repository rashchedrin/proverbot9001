#![feature(vec_remove_item)]
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use std::cmp::min;
use std::fs::File;

mod scraped_data;
use scraped_data::*;
mod features;
use features::{context_features, PickleableTokenMap, TokenMap, VEC_FEATURES_SIZE};
mod models;
use models::features_polyarg_predictor::*;
mod paren_util;
mod tokenizer;

#[macro_use]
extern crate lazy_static;

extern crate rayon;
#[pymodule]
fn dataloader(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "features_to_total_distances_tensors")]
    fn parallel_features_to_total_distances_tensors(
        py: Python,
        args: DataloaderArgs,
        filename: String,
    ) -> PyResult<(
        TokenMap,
        LongTensor2D,
        FloatTensor2D,
        FloatTensor2D,
        Vec<i64>,
        i64,
    )> {
        py.allow_threads(move || features_to_total_distances_tensors(args, filename, None))
    }
    #[pyfn(m, "features_to_total_distances_tensors_with_map")]
    fn parallel_features_to_total_distances_tensors_with_map(
        py: Python,
        args: DataloaderArgs,
        filename: String,
        map: TokenMap,
    ) -> PyResult<(
        TokenMap,
        LongTensor2D,
        FloatTensor2D,
        FloatTensor2D,
        Vec<i64>,
        i64,
    )> {
        py.allow_threads(move || features_to_total_distances_tensors(args, filename, Some(map)))
    }
    #[pyfn(m, "features_polyarg_tensors")]
    fn parallel_features_polyarg_tensors_py(
        py: Python,
        args: DataloaderArgs,
        filename: String,
    ) -> PyResult<(
        PickleableFPAMetadata,
        (
            LongUnpaddedTensor3D,
            FloatUnpaddedTensor3D,
            LongTensor1D,
            LongTensor2D,
            LongTensor2D,
            FloatTensor2D,
            LongTensor1D,
            LongTensor1D,
        ),
        (Vec<i64>, i64),
    )> {
        py.allow_threads(move || features_polyarg_tensors(args, filename, None))
    }
    #[pyfn(m, "features_polyarg_tensors_with_meta")]
    fn parallel_features_polyarg_tensors_with_meta(
        py: Python,
        args: DataloaderArgs,
        filename: String,
        meta: PickleableFPAMetadata,
    ) -> PyResult<(
        PickleableFPAMetadata,
        (
            LongUnpaddedTensor3D,
            FloatUnpaddedTensor3D,
            LongTensor1D,
            LongTensor2D,
            LongTensor2D,
            FloatTensor2D,
            LongTensor1D,
            LongTensor1D,
        ),
        (Vec<i64>, i64),
    )> {
        py.allow_threads(move || features_polyarg_tensors(args, filename, Some(meta)))
    }
    #[pyfn(m, "sample_fpa")]
    fn sample_fpa_py(
        _py: Python,
        args: DataloaderArgs,
        metadata: PickleableFPAMetadata,
        relevant_lemmas: Vec<String>,
        prev_tactics: Vec<String>,
        hypotheses: Vec<String>,
        goal: String,
    ) -> (
        LongUnpaddedTensor3D,
        FloatUnpaddedTensor3D,
        LongTensor1D,
        LongTensor2D,
        LongTensor2D,
        FloatTensor2D,
    ) {
        sample_fpa(
            args,
            metadata,
            relevant_lemmas,
            prev_tactics,
            hypotheses,
            goal,
        )
    }
    #[pyfn(m, "decode_fpa_result")]
    fn decode_fpa_result_py(
        _py: Python,
        args: DataloaderArgs,
        metadata: PickleableFPAMetadata,
        hyps: Vec<String>,
        goal: &str,
        tac_idx: i64,
        arg_idx: i64,
    ) -> String {
        decode_fpa_result(args, metadata, hyps, goal, tac_idx, arg_idx)
    }
    #[pyfn(m, "get_num_tokens")]
    fn get_num_tokens(_py: Python, metadata: PickleableFPAMetadata) -> i64 {
        let (_indexer, tokenizer, _ftmap) = fpa_metadata_from_pickleable(metadata);
        tokenizer.num_tokens()
    }
    #[pyfn(m, "get_num_indices")]
    fn get_num_indices(_py: Python, metadata: PickleableFPAMetadata) -> i64 {
        let (indexer, _tokenizer, _ftmap) = fpa_metadata_from_pickleable(metadata);
        indexer.num_indices()
    }

    m.add_wrapped(wrap_pyfunction!(features_vocab_sizes))?;
    m.add_wrapped(wrap_pyfunction!(tmap_from_picklable))?;
    m.add_wrapped(wrap_pyfunction!(tmap_to_picklable))?;
    m.add_wrapped(wrap_pyfunction!(sample_context_features))?;
    m.add_class::<TokenMap>()?;
    m.add_class::<DataloaderArgs>()?;
    Ok(())
}
fn features_to_total_distances_tensors(
    args: DataloaderArgs,
    filename: String,
    map: Option<TokenMap>,
) -> PyResult<(
    TokenMap,
    LongTensor2D,
    FloatTensor2D,
    FloatTensor2D,
    Vec<i64>,
    i64,
)> {
    match File::open(filename) {
        Result::Ok(file) => {
            let scraped = scraped_from_file(file).collect();
            let distanced = tactic_distances(scraped);
            let (tactics, distances): (Vec<ScrapedTactic>, Vec<usize>) =
                distanced.into_iter().unzip();
            let outputs = normalize_distances(args.max_distance, distances)
                .into_iter()
                .map(|distance| vec![distance])
                .collect();
            let tmap = match map {
                Some(m) => m,
                None => TokenMap::initialize(&tactics, args.num_keywords),
            };
            let (word_features, float_features) = context_features(args, &tmap, tactics);
            let word_features_sizes = tmap.word_features_sizes();

            Ok((
                tmap,
                word_features,
                float_features,
                outputs,
                word_features_sizes,
                VEC_FEATURES_SIZE,
            ))
        }
        Result::Err(_err) => Err(PyErr::new::<exceptions::TypeError, _>(
            "Failed to open file",
        )),
    }
}
#[pyfunction]
fn sample_context_features(
    args: DataloaderArgs,
    tmap: &TokenMap,
    relevant_lemmas: Vec<String>,
    prev_tactics: Vec<String>,
    hypotheses: Vec<String>,
    goal: String,
) -> (LongTensor1D, FloatTensor1D) {
    features::sample_context_features(args, tmap, relevant_lemmas, prev_tactics, hypotheses, goal)
}

#[pyfunction]
fn features_vocab_sizes(tmap: TokenMap) -> (Vec<i64>, i64) {
    (tmap.word_features_sizes(), VEC_FEATURES_SIZE)
}

#[pyfunction]
fn tmap_to_picklable(tmap: TokenMap) -> PickleableTokenMap {
    tmap.to_dicts()
}

#[pyfunction]
fn tmap_from_picklable(picklable: PickleableTokenMap) -> TokenMap {
    TokenMap::from_dicts(picklable)
}

fn normalize_distances(max_distance: usize, distances: Vec<usize>) -> Vec<f64> {
    distances
        .into_iter()
        .map(|x| (min(x, max_distance) as f64) / (max_distance as f64))
        .collect()
}

fn tactic_distances(scraped_data: Vec<ScrapedData>) -> Vec<(ScrapedTactic, usize)> {
    let mut in_proof = false;
    let mut interaction_buffer = Vec::new();
    let mut blocks = Vec::new();

    for interaction in scraped_data {
        match interaction {
            ScrapedData::Tactic(tac) => {
                if !in_proof {
                    interaction_buffer.clear();
                    in_proof = true;
                }
                interaction_buffer.push(tac)
            }
            ScrapedData::Vernac(_cmd) => {
                if in_proof {
                    blocks.push(interaction_buffer.clone());
                    in_proof = false;
                }
            }
        }
    }

    let mut result = Vec::new();

    for block in blocks {
        let mut distanced_block: Vec<(ScrapedTactic, usize)> = label_block_distances(block);
        result.append(&mut distanced_block);
    }
    return result;
}

fn label_block_distances(block: Vec<ScrapedTactic>) -> Vec<(ScrapedTactic, usize)> {
    let mut path_segments: Vec<Vec<ScrapedTactic>> = vec![Vec::new()];
    let mut closed_distances: Vec<usize> = vec![0, 0];
    let mut finished_segments: Vec<Vec<(ScrapedTactic, usize)>> = vec![Vec::new(), Vec::new()];

    let close_goal =
        |path_segments: &mut Vec<Vec<ScrapedTactic>>,
         closed_distances: &mut Vec<usize>,
         finished_segments: &mut Vec<Vec<(ScrapedTactic, usize)>>| {
            let last_segment = path_segments.pop().expect("Not enough path segments");
            let last_segment_len = last_segment.len();
            let last_distance = closed_distances.pop().expect("Not enough closed distances");
            let mut closed_tacs: Vec<(ScrapedTactic, usize)> = last_segment
                .into_iter()
                .rev()
                .zip((1 + last_distance)..)
                .collect::<Vec<(ScrapedTactic, usize)>>()
                .into_iter()
                .rev()
                .collect();

            let mut already_closed_tacs = finished_segments
                .pop()
                .expect("Not enough finished segments");
            let last_finished_segment = finished_segments
                .last_mut()
                .expect("Not enough finished segments");
            last_finished_segment.append(&mut closed_tacs);
            last_finished_segment.append(&mut already_closed_tacs);
            let next_last_distance = closed_distances
                .last_mut()
                .expect("Not enough closed distances");
            *next_last_distance += last_distance + last_segment_len
        };

    for interaction in block.into_iter() {
        let trimmed_tac = interaction.tactic.trim();
        if trimmed_tac == "{" {
            path_segments.push(Vec::new());
            closed_distances.push(0);
            finished_segments.push(Vec::new());
        } else if trimmed_tac == "}" {
            close_goal(
                &mut path_segments,
                &mut closed_distances,
                &mut finished_segments,
            );
        } else if trimmed_tac == "Qed." {
            close_goal(
                &mut path_segments,
                &mut closed_distances,
                &mut finished_segments,
            );
            {
                let last_finished_segment = finished_segments
                    .last_mut()
                    .expect("Not enougn finished segments");
                last_finished_segment.push((interaction, 0));
            }
            return finished_segments.pop().unwrap();
        } else {
            let last_path_segment = path_segments
                .last_mut()
                .expect("Not enougn finished segments");
            last_path_segment.push(interaction);
        }
    }
    assert_eq!(path_segments.len(), 1);
    close_goal(
        &mut path_segments,
        &mut closed_distances,
        &mut finished_segments,
    );
    assert_eq!(finished_segments.len(), 1);
    finished_segments
        .pop()
        .expect("Not enough finished segments")
}
