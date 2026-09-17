use std::f64::consts::E;

fn main() {
    // layers: input -> hidden -> output. 3 nodes each

    // Layer 1: input
    let (i1, i2, i3) = (0.9, 0.1, 0.8);

    // Layer 2: hidden weights
    let (hw11, hw21, hw31) = (0.9, 0.3, 0.4);
    let (hw12, hw22, hw32) = (0.2, 0.8, 0.2);
    let (hw13, hw23, hw33) = (0.1, 0.5, 0.6);

    // Layer 3: output weights
    let (ow11, ow21, ow31) = (0.3, 0.7, 0.5);
    let (ow12, ow22, ow32) = (0.6, 0.5, 0.2);
    let (ow13, ow23, ow33) = (0.8, 0.1, 0.9);

    let inputs = vec![i1, i2, i3];
    let h_weights = vec![vec![hw11, hw21, hw31], vec![hw12, hw22, hw32], vec![hw13, hw23, hw33]];
    let o_weights = vec![vec![ow11, ow21, ow31], vec![ow12, ow22, ow32], vec![ow13, ow23, ow33]];

    let mut output;
    output = forward(inputs, h_weights);
    println!("Output layer 2: {:?}", output);

    output = forward(output, o_weights);
    println!("Output layer 3: {:?}", output);
}

fn forward(inputs:Vec<f64>, weights:Vec<Vec<f64>>) -> Vec<f64>{
    println!("inputs len: {}", inputs.len());
    let mut output:Vec<f64> = vec![0.; inputs.len()];

    println!("outputs len: {}", output.len());

    for (x, w) in weights.iter().enumerate() {
        for (y, i) in inputs.iter().enumerate() {
            output[x] += i * weights[x][y];
            println!("I: {} * W: {} = {}", i, weights[x][y], output[x]);
        }

        println!("Output pre-sigm: {:?}", output[x]);

        output[x] = apply_sigmoid(output[x]);
    }

    return output;
}

fn apply_sigmoid(input:f64) -> f64 { 1./(1. + f64::powf(E, -input)) }
