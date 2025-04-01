use bullet_lib::{
  default::inputs::{Chess768, ChessBucketsMirrored}, nn::{
    optimiser::{AdamWOptimiser, AdamWParams}, Activation, ExecutionContext, Graph, InitSettings, NetworkBuilder, NetworkBuilderNode, Node, Shape
  }, optimiser::Optimiser, trainer::{
    default::{inputs, loader, outputs, Trainer},
    save::{Layout, QuantTarget, SavedFormat},
    schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
    settings::LocalSettings,
  }, NetworkTrainer
};
use bulletformat::ChessBoard;

// Parameters
const HL: usize = 128;

// (HL=128, SYM=true) has approximately the same number of parameters as (HL=64,SYM=false)
// (SYM=true only works with MyInputs, SYM=false should give the same results with either ChessBucketsMirrored and MyInputs).

// type InputFeatures = ChessBucketsMirrored;
// const SYM: bool = false;

type InputFeatures = MyInputs;
const SYM: bool = true;

const NET: &str = "train-128-sym";

// Constants
const CHESS_INPUTS: usize = 768;

pub fn map_square(sq: usize) -> usize {
  let a = sq / 8;
  let mut b = sq % 8;
  let rb = b >= 4;
  b %= 4;
  if rb { b = 3-b }
  ((rb as usize) * 4 + b) * 8 + a
}


#[derive(Clone, Copy, Debug, Default)]
pub struct MyInputs;

// Same as ChessBucketsMirrored, but permutes the inputs so that (i + 384) is horizontally symmetric to i (for i < 384).
impl inputs::SparseInputType for MyInputs {
  type RequiredDataType = ChessBoard;

  fn num_inputs(&self) -> usize {
    CHESS_INPUTS
  }

  fn max_active(&self) -> usize {
    32
  }

  fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
    let get = |ksq| if ksq % 8 > 3 { 7 } else { 0 };
    let stm_flip = get(pos.our_ksq());
    let ntm_flip = get(pos.opp_ksq());

    for (piece, square) in pos.into_iter() {
      let c = usize::from(piece & 8 > 0);
      let pc = usize::from(piece & 7);
      let sq = usize::from(square);
      
      let stm = [pc, 6+pc][c] + 12*map_square(sq ^ stm_flip);
      let ntm = [6+pc, pc][c] + 12*map_square(sq ^ ntm_flip ^ 56);
      f(stm, ntm);
    }
  }

  fn shorthand(&self) -> String { "".to_string() }

  fn description(&self) -> String { "".to_string() }
}

type OutputBuckets = outputs::MaterialCount<8>;

fn main() {
  let inputs = InputFeatures::default();
  let output_buckets = OutputBuckets::default();

  let (mut graph, output_node) = build_network();

  graph.get_weights_mut("l0w").seed_random(0.0, 1.0 / (CHESS_INPUTS as f32).sqrt(), true);
  if SYM {
    graph.get_weights_mut("l0b_").seed_random(0.0, 1.0 / (CHESS_INPUTS as f32).sqrt(), true);
  }else{
    graph.get_weights_mut("l0b").seed_random(0.0, 1.0 / (CHESS_INPUTS as f32).sqrt(), true);
  }
  graph.get_weights_mut("l1w").seed_random(0.0, 1.0 / (HL as f32).sqrt(), true);
  graph.get_weights_mut("l1b").seed_random(0.0, 1.0 / (HL as f32).sqrt(), true);
  
  let mut trainer = Trainer::<AdamWOptimiser, _, _>::new(
    graph,
    output_node,
    AdamWParams::default(),
    inputs,
    output_buckets,
    vec![
      SavedFormat::new("l0w", QuantTarget::Float, Layout::Normal),
      SavedFormat::new(if SYM { "l0b_" } else { "l0b" }, QuantTarget::Float, Layout::Normal),
      SavedFormat::new("l1w", QuantTarget::Float, Layout::Normal),
      SavedFormat::new("l1b", QuantTarget::Float, Layout::Normal),
    ],
    true,
  );

  if SYM {
    trainer.optimiser_mut().set_params_for_weight
      ("l0b",
       AdamWParams { decay: 0.1, beta1: 0.9, beta2: 0.999, min_weight: -0.0, max_weight: 0.0 });
  }

  let sbs = 40;
  
  let schedule = TrainingSchedule {
    net_id: NET.to_string(),
    eval_scale: 400.0,
    steps: TrainingSteps {
      batch_size: 16_384,
      batches_per_superbatch: 6104,
      start_superbatch: 1,
      end_superbatch: sbs,
    },
    wdl_scheduler: wdl::ConstantWDL { value: 0.5 },
    lr_scheduler: lr::CosineDecayLR { initial_lr: 0.001, final_lr: 0.00002, final_superbatch: sbs },
    save_rate: 1,
  };

  let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 512 };

  let data_loader = loader::DirectSequentialDataLoader::new(&[
    "shuffled.bin"
  ]);

  trainer.run(&schedule, &settings, &data_loader);

  let eval = 400.0 * trainer.eval("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 0 | 0.0");
  println!("Eval: {eval:.3}cp");
}

pub fn shuffle_h(a: NetworkBuilderNode<'_>) -> NetworkBuilderNode<'_> {
  let a0 = a.slice_rows(0, HL/2);
  let a1 = a.slice_rows(HL/2, HL);
  a1.concat(a0)
}

fn build_network() -> (Graph, Node) {
  let builder = NetworkBuilder::default();

  // inputs
  let stm = builder.new_input("stm", Shape::new(CHESS_INPUTS, 1));
  let nstm = builder.new_input("nstm", Shape::new(CHESS_INPUTS, 1));
  let targets = builder.new_input("targets", Shape::new(1, 1));
  let buckets = builder.new_input("buckets", Shape::new(8, 1));

  // trainable weights
  let l0 = builder.new_affine("l0", if SYM { CHESS_INPUTS / 2 } else { CHESS_INPUTS }, HL);
  let l1 = builder.new_affine("l1", 2 * HL, 8);

  // inference
  let mut out;
  if SYM {
    let l0b = builder.new_weights("l0b_", Shape::new(HL, 1), InitSettings::Zeroed);
    
    let stm0 = stm.slice_rows(0, CHESS_INPUTS/2);
    let stm1 = stm.slice_rows(CHESS_INPUTS/2, CHESS_INPUTS);
    let nstm0 = nstm.slice_rows(0, CHESS_INPUTS/2);
    let nstm1 = nstm.slice_rows(CHESS_INPUTS/2, CHESS_INPUTS);
    
    let out0 =
      l0.forward(stm0) +
      shuffle_h(l0.forward(stm1)) +
      l0b;
    let out1 =
      l0.forward(nstm0) +
      shuffle_h(l0.forward(nstm1)) +
      l0b;

    out = out0.concat(out1);
  }else{
    let out0 = l0.forward(stm);
    let out1 = l0.forward(nstm);
    out = out0.concat(out1);
  }
  
  out = out.activate(Activation::SCReLU);
  out = l1.forward(out).select(buckets);
  let pred = out.activate(Activation::Sigmoid);
  pred.mse(targets);

  let output_node = out.node();
  (builder.build(ExecutionContext::default()), output_node)
}
