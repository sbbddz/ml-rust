use core::fmt;

use crate::get_random_float;

pub struct Matrix {
    rows: usize,
    cols: usize,
    imp: Vec<f32>,
}

impl Matrix {
    pub fn new(cols: usize, rows: usize) -> Matrix {
        let mat = Matrix {
            rows,
            cols,
            imp: vec![0f32; cols * rows],
        };
        mat
    }

    pub fn dot(&self, matrix: &Matrix) -> Matrix {
        if self.cols != matrix.rows {
            panic!("You cannot multiply a matrix which its cols does not match the rows of the other matrix");
        }

        let mut new_matrix = Matrix::new(matrix.cols, self.rows);

        let n = self.cols;

        for i in 0..new_matrix.rows {
            for j in 0..new_matrix.cols {
                for k in 0..n {
                    *new_matrix.at_mut(i, j).unwrap() +=
                        self.at(i, k).unwrap() * matrix.at(k, j).unwrap();
                }
            }
        }
        new_matrix
    }

    pub fn sum(&mut self, matrix: &Matrix) -> () {
        for i in 0..self.rows {
            for j in 0..self.cols {
                *self.at_mut(i, j).unwrap() += matrix.at(i, j).unwrap();
            }
        }
    }

    pub fn at(&self, x: usize, y: usize) -> Option<&f32> {
        self.imp.get(x * self.cols + y)
    }

    pub fn at_mut(&mut self, x: usize, y: usize) -> Option<&mut f32> {
        self.imp.get_mut(x * self.cols + y)
    }

    pub fn rand(&mut self, min: f32, max: f32) -> () {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.imp[i * self.cols + j] = get_random_float() * (max - min) + min;
            }
        }
    }

    pub fn fill(&mut self, value: f32) -> () {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.imp[i * self.cols + j] = value;
            }
        }
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut string = String::new();

        for i in 0..self.rows {
            for j in 0..self.cols {
                let value = format!("{:?}", self.at(i, j));
                string.push_str(&value);
                string.push_str("\t");
            }
            string.push_str("\n");
        }

        write!(f, "{}", string)
    }
}

#[cfg(test)]
mod tests {
    use super::Matrix;

    #[test]
    fn void() {
        let mut matrix = Matrix::new(1, 2);
        let mut another_matrix = Matrix::new(2, 1);
        matrix.rand(0f32, 10f32);
        another_matrix.rand(0f32, 10f32);
        println!("matrix_1:\n {}", matrix);
        println!("matrix_2:\n {}", another_matrix);
        println!("dot:\n {}", matrix.dot(&another_matrix));
    }
}
