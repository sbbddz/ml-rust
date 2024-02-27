mod boolean_gates;
mod double;

#[derive(Clone, Copy, Debug)]
struct Neuron {
    weight_1: f32,
    weight_2: f32,
    bias: f32,
}

type Model = [Neuron; 3];

fn sigmoid(value: f32) -> f32 {
    return 1f32 / (1f32 + value.exp());
}

fn forward(model: &Model, x1: f32, x2: f32) -> f32 {
    let a: f32 = sigmoid(model[0].weight_1 * x1 + model[0].weight_2 * x2 + model[0].bias);
    let b: f32 = sigmoid(model[1].weight_1 * x1 + model[1].weight_2 * x2 + model[1].bias);
    sigmoid(a * model[2].weight_1 + b * model[2].weight_2 + model[2].bias)
}

fn loss_fn(train_set: &Vec<(i32, i32, i32)>, model: Model) -> f32 {
    let mut result = 0f32;
    for t_value in train_set {
        let output = forward(&model, t_value.0 as f32, t_value.1 as f32);
        let diff = output - t_value.2 as f32;
        result += diff * diff;
    }
    return result;
}

fn get_random_float() -> f32 {
    rand::random()
}

fn create_random_neuron() -> Neuron {
    Neuron {
        weight_1: get_random_float(),
        weight_2: get_random_float(),
        bias: get_random_float(),
    }
}

fn create_random_model() -> Model {
    [
        create_random_neuron(),
        create_random_neuron(),
        create_random_neuron(),
    ]
}

fn apply_neuron_diff(neuron: Neuron, diff: Neuron) -> Neuron {
    let rate = 0.01;
    Neuron {
        weight_1: neuron.weight_1 - rate * diff.weight_1,
        weight_2: neuron.weight_2 - rate * diff.weight_2,
        bias: neuron.bias - rate * diff.bias,
    }
}

fn numerical_diff_on_neuron(
    neuron_index: usize,
    model: Model,
    train_set: &Vec<(i32, i32, i32)>,
    loss_value: f32,
) -> Neuron {
    let eps = 0.01;

    let mut model_dw1 = model.clone();
    model_dw1[neuron_index].weight_1 += eps;
    let dw1 = (loss_fn(&train_set, model_dw1) - loss_value) / eps;

    let mut model_dw2 = model.clone();
    model_dw2[neuron_index].weight_2 += eps;
    let dw2 = (loss_fn(&train_set, model_dw2) - loss_value) / eps;

    let mut model_dwb = model.clone();
    model_dwb[neuron_index].bias += eps;
    let dwb = (loss_fn(&train_set, model_dwb) - loss_value) / eps;

    Neuron {
        weight_1: dw1,
        weight_2: dw2,
        bias: dwb,
    }
}

fn main() {
    let xor_train_set: Vec<(i32, i32, i32)> = vec![(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)];
    let mut model = create_random_model();

    for _ in 0..200*2000 {
        let loss = loss_fn(&xor_train_set, model);
        println!("loss = {}", loss);

        let mut new_model: [Neuron; 3] = model.clone();

        for (i, _) in model.iter().enumerate() {
            let neuron_diff = numerical_diff_on_neuron(i, model, &xor_train_set, loss);
            new_model[i] = apply_neuron_diff(model[i], neuron_diff);
        }

        model = new_model;

        println!("loss = {}", loss_fn(&xor_train_set, model));
    }

    for x in 0..2 {
        for y in 0..2 {
            println!("{} ^ {} = {}", x, y, forward(&model, x as f32, y as f32));
        }
    }
}
