use super::{
    matrix::*,
    errors::MatrixError
};

pub fn transpose<const R: usize, const C: usize>(
    matrix: &Matrix<R, C>
) -> Result<Matrix<C, R>, MatrixError> {
    let mut transposed: Vec<f32> = Vec::new();

    for col in 0..C {
        for row in 0..R {
            transposed.push(matrix.get(row, col)?);
        }
    }

    Matrix::<C, R>::new(transposed)
}

pub fn dot<const A: usize, const B: usize, const C: usize>(
    lhs: &Matrix<A, B>,
    rhs: &Matrix<B, C>
) -> Result<Matrix<A, C>, MatrixError> {
    // TODO: better algo (Solvay Strassen Algorithm)
    // https://www.baeldung.com/cs/matrix-multiplication-algorithms

    let mut data: Vec<f32> = Vec::new();
    for row in 0..A {
        for col in 0..C {
            let mut val = 0.0;
            for k in 0..B {
                let lhs_val = lhs.get(row, k)?;
                let rhs_val = rhs.get(k, col)?;

                val += lhs_val * rhs_val;
            }

            data.push(val);
        }
    }

    Matrix::<A, C>::new(data)
}

pub fn vector_dot<const A: usize>(lhs: &Vector<A>, rhs: &Vector<A>) -> Result<f32, MatrixError> {
    let transposed_lhs = transpose(lhs)?;

    let result = dot(&transposed_lhs, rhs)?;

    Ok(result.raw_data()[0])
}

pub fn vector_length<const A: usize>(vector: &Vector<A>) -> Result<f32, MatrixError> {
    let mut length = 0.0;
    for val in vector.raw_data() {
        length += val.powf(2.0);
    }

    Ok(length.sqrt())
}

pub fn vector_unit<const A: usize>(vector: &Vector<A>) -> Result<Vector<A>, MatrixError> {
    let length = vector_length(vector)?;
    let raw_data = vector.raw_data();

    let mut data: Vec<f32> = Vec::new();
    for val in raw_data {
        data.push(val / length);
    }

    Vector::<A>::new(data)
}

pub fn vector_cross(lhs: &Vector3, rhs: &Vector3) -> Result<Vector3, MatrixError> {
    let a = lhs.raw_data();
    let b = rhs.raw_data();

    Vector3::new(vec![
        (a[1] * b[2]) - (a[2] * b[1]),
        (a[2] * b[0]) - (a[0] * b[2]),
        (a[0] * b[1]) - (a[1] * b[0])
    ])
}
