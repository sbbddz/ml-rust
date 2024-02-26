fn get_random_float() -> f32 {
    rand::random::<f32>() * 10f32
}

type Model = f32;

fn loss_fn(input_data: &Vec<(i32, i32)>, model: Model, bias: f32) -> f32 {
    let mut result = 0f32;
    for t_data in input_data {
        let input = t_data.0;
        let expected = t_data.1;

        let output = input as f32 * model;
        result += (output - expected as f32) * (output - expected as f32); // as the difference between output and expected grows,
        // it also does our result, if result is nearer to 0 our model is very nice
    }
    result
}

fn double() {
    let input_data = vec![(1, 2), (2, 4), (4, 8), (6, 12), (12, 24)];
    let mut weight = get_random_float();
    let mut bias = get_random_float();

    let eps = 0.001;
    let rate = 0.001;

    println!("{}", loss_fn(&input_data, weight + eps, bias));
    for _ in 0..300 {
        let diffcost = (loss_fn(&input_data, weight + eps, bias) - loss_fn(&input_data, weight, bias)) / eps;
        let biascost = (loss_fn(&input_data, weight, bias + eps) - loss_fn(&input_data, weight, bias)) / eps;
        weight -= rate*diffcost;
        bias -= rate*biascost;
        println!("{}", loss_fn(&input_data, weight, bias));
    }
    println!("weight: {}, bias: {}", weight, bias);
}
