mod boolean_gates;
mod double;

#[derive(Clone, Copy, Debug)]
struct Model {
    or_w1: f32,
    or_w2: f32,
    or_b: f32,
    and_w1: f32,
    and_w2: f32,
    and_b: f32,
    nand_w1: f32,
    nand_w2: f32,
    nand_b: f32,
}

fn sigmoid(value: f32) -> f32 {
    return 1f32 / (1f32 + value.exp());
}

fn forward(model: &Model, x1: f32, x2: f32) -> f32 {
    let a: f32 = sigmoid(model.or_w1 * x1 + model.or_w2 * x2 + model.or_b);
    let b: f32 = sigmoid(model.nand_w1 * x1 + model.nand_w2 * x2 + model.nand_b);
    sigmoid(a * model.and_w1 + b * model.and_w2 + model.and_b)
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

fn create_random_model() -> Model {
    Model {
        or_w1: get_random_float(),
        or_w2: get_random_float(),
        or_b: get_random_float(),
        and_w1: get_random_float(),
        and_w2: get_random_float(),
        and_b: get_random_float(),
        nand_w1: get_random_float(),
        nand_w2: get_random_float(),
        nand_b: get_random_float(),
    }
}

fn main() {
    let xor_train_set: Vec<(i32, i32, i32)> = vec![(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)];
    let mut model = create_random_model();

    let eps = 0.1;
    let rate = 0.1;

    for _ in 0..200 * 200 {
        let loss = loss_fn(&xor_train_set, model);
        println!("loss = {}", loss);

        let diff_or_w1 = (loss_fn(
            &xor_train_set,
            Model {
                or_w1: model.or_w1 + eps,
                ..model
            },
        ) - loss)
            / eps;
        let diff_or_w2 = (loss_fn(
            &xor_train_set,
            Model {
                or_w2: model.or_w2 + eps,
                ..model
            },
        ) - loss)
            / eps;
        let diff_or_b = (loss_fn(
            &xor_train_set,
            Model {
                or_b: model.or_b + eps,
                ..model
            },
        ) - loss)
            / eps;
        let diff_and_w1 = (loss_fn(
            &xor_train_set,
            Model {
                and_w1: model.and_w1 + eps,
                ..model
            },
        ) - loss)
            / eps;
        let diff_and_w2 = (loss_fn(
            &xor_train_set,
            Model {
                and_w2: model.and_w2 + eps,
                ..model
            },
        ) - loss)
            / eps;
        let diff_and_b = (loss_fn(
            &xor_train_set,
            Model {
                and_b: model.and_b + eps,
                ..model
            },
        ) - loss)
            / eps;
        let diff_nand_w1 = (loss_fn(
            &xor_train_set,
            Model {
                nand_w1: model.nand_w1 + eps,
                ..model
            },
        ) - loss)
            / eps;
        let diff_nand_w2 = (loss_fn(
            &xor_train_set,
            Model {
                nand_w2: model.nand_w2 + eps,
                ..model
            },
        ) - loss)
            / eps;
        let diff_nand_b = (loss_fn(
            &xor_train_set,
            Model {
                nand_b: model.nand_b + eps,
                ..model
            },
        ) - loss)
            / eps;

        model = Model {
            or_w1: model.or_w1 - rate * diff_or_w1,
            or_w2: model.or_w2 - rate * diff_or_w2,
            or_b: model.or_b - rate * diff_or_b,
            and_w1: model.and_w1 - rate * diff_and_w1,
            and_w2: model.and_w2 - rate * diff_and_w2,
            and_b: model.and_b - rate * diff_and_b,
            nand_w1: model.nand_w1 - rate * diff_nand_w1,
            nand_w2: model.nand_w2 - rate * diff_nand_w2,
            nand_b: model.nand_b - rate * diff_nand_b,
        };

        println!("loss = {}", loss_fn(&xor_train_set, model));
    }

    for x in 0..2 {
        for y in 0..2 {
            println!("{} ^ {} = {}", x, y, forward(&model, x as f32, y as f32));
        }
    }
}
