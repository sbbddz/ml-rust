use rand::{rngs::StdRng, Rng, SeedableRng};

/// w1, w2, b
type Model = (f32, f32, f32);

fn sigmoid(value: f32) -> f32 {
    return 1f32 / (1f32 + value.exp());
}

fn loss_fn(input_data: &Vec<(i32, i32, i32)>, model: Model) -> f32 {
    let mut result = 0f32;
    for t_data in input_data {
        let expected = t_data.2;
        let output = sigmoid(t_data.0 as f32 * model.0 + t_data.1 as f32 * model.1 + model.2);
        let diff = output - expected as f32;
        result += diff * diff
    }
    result
}

fn get_random_float() -> f32 {
    let mut r = StdRng::seed_from_u64(10);
    r.gen::<f32>()
}

fn main() {
    let or_train_set = vec![(0, 0, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)];
    let and_train_set = vec![(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 1)];
    let nand_train_set = vec![(0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 0)];

    print_result(train_input(&or_train_set));
    print_result(train_input(&and_train_set));
    print_result(train_input(&nand_train_set));
}

fn train_input(train_set: &Vec<(i32, i32, i32)>) -> Model {
    let mut model = (get_random_float(), get_random_float(), get_random_float());
    let eps = 0.01;
    let rate = 0.01;

    for _ in 0..100 * 1000 {
        let loss = loss_fn(&train_set, model);
        // Numerical differentiation
        let diff_w1 = (loss_fn(&train_set, (model.0 + eps, model.1, model.2)) - loss) / eps;
        let diff_w2 = (loss_fn(&train_set, (model.0, model.1 + eps, model.2)) - loss) / eps;
        let diff_bias = (loss_fn(&train_set, (model.0, model.1, model.2 + eps)) - loss) / eps;
        model = (
            model.0 - rate * diff_w1,
            model.1 - rate * diff_w2,
            model.2 - rate * diff_bias,
        );
        // println!("loss = {}", loss_fn(&train_set, model));
    }

    println!("model = {:?}", model);
    model
}

fn print_result(model: Model) {
    for i in 0..2 {
        for j in 0..2 {
            println!(
                "{}\t\t{} = {}",
                i,
                j,
                sigmoid(i as f32 * model.0 + j as f32 * model.1 + model.2)
            );
        }
    }
}
