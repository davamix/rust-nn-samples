use std::f64::consts::E;

fn main() {
    // layers: input -> hidden -> output. 3 nodes each
    let (i1, i2, i3) = (0.9, 0.1, 0.8);
    let (w11, w21, w31) = (0.9, 0.3, 0.4);
    let (w12, w22, w32) = (0.2, 0.8, 0.2);
    let (w13, w23, w33) = (0.1, 0.5, 0.6);

    let inputs = vec![i1, i2, i3];
    let weights = vec![vec![w11, w21, w31], vec![w12, w22, w32], vec![w13, w23, w33]];

    let output = forward(inputs, weights);

    println!("Output: {:?}", output);
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
