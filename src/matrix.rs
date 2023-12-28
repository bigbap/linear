use std::ops;

#[derive(Debug, PartialEq, thiserror::Error)]
pub enum MatrixError {
    #[error("cannot create a matrix with empty data")]
    EmptyData,

    #[error("the matrix is not a vector")]
    NotAVector,

    #[error("location is out of bounds")]
    OutOfBounds,

    #[error("the provided data does not match the matrix dimensions")]
    TheDataDoesNotMatchTheDimensions,

    #[error("lhs and rhs matrices must have the same dimensions")]
    MatricesMustHaveSameDimensions,

    #[error("lhs and rhs matrices do not have matching dimensions for dot product")]
    MatrixDimensionsDoNotMatchForDotProduct,
    
    #[error("lhs and rhs must be vector_3 for cross product")]
    VectorMustBe3DForCrossProduct
}

#[derive(Debug, PartialEq)]
pub struct Matrix {
    data: Vec<f32>,
    rows: usize,
    cols: usize
}

impl Matrix {
    pub fn new(data: Vec<Vec<f32>>) -> Result<Self, MatrixError> {
        if data.is_empty() {
            return Err(MatrixError::EmptyData);
        }

        let rows = data.len();
        let cols = data[0].len();
        let data: Vec<f32> = data.into_iter().flatten().collect();

        if data.len() != rows * cols {
            return Err(MatrixError::TheDataDoesNotMatchTheDimensions);
        }

        Ok(Self {
            data,
            rows,
            cols
        })
    }

    pub fn from_flat_vec(data: Vec<f32>, rows: usize, cols: usize) -> Result<Self, MatrixError> {
        if data.len() != rows * cols {
            return Err(MatrixError::TheDataDoesNotMatchTheDimensions);
        }

        Ok(Self {
            data,
            rows,
            cols
        })
    }

    pub fn vector(data: Vec<f32>) -> Result<Self, MatrixError> {
        Self::from_flat_vec(data.clone(), data.len(), 1)
    }

    pub fn vector_2(data: (f32, f32)) -> Result<Self, MatrixError> {
        Self::vector(vec![data.0, data.1])
    }
    pub fn vector_3(data: (f32, f32, f32)) -> Result<Self, MatrixError> {
        Self::vector(vec![data.0, data.1, data.2])
    }
    pub fn vector_4(data: (f32, f32, f32, f32)) -> Result<Self, MatrixError> {
        Self::vector(vec![data.0, data.1, data.2, data.3])
    }

    pub fn vector_length(&self) -> Result<f32, MatrixError> {
        self.assert_vector()?;

        let mut length = 0.0;
        for val in self.raw_data() {
            length += val.powf(2.0);
        }

        Ok(length.sqrt())
    }

    pub fn vector_unit(&self) -> Result<Matrix, MatrixError> {
        let length = self.vector_length()?;
        let raw_data = self.raw_data();

        let mut data: Vec<f32> = Vec::new();
        for val in raw_data {
            data.push(val / length);
        }

        Self::vector(data)
    }

    pub fn raw_data(&self) -> &[f32] {
        &self.data
    }

    pub fn get(&self, row: usize, col: usize) -> Result<f32, MatrixError> {
        get(&self.data, self.cols, (row, col))
    }

    pub fn dot(&self, rhs: &Self) -> Result<Self, MatrixError> {
        let (result, rows, cols) = dot(
            (self.raw_data().to_vec(), self.rows, self.cols),
            (rhs.raw_data().to_vec(), rhs.rows, rhs.cols)
        )?;

        Self::from_flat_vec(result, rows, cols)
    }

    pub fn vector_dot(&self, rhs: &Self) -> Result<f32, MatrixError> {
        self.assert_vector()?;
        rhs.assert_vector()?;

        Ok(self.dot(rhs)?.raw_data()[0])
    }

    pub fn vector_cross(&self, rhs: &Self) -> Result<Matrix, MatrixError> {
        self.assert_vector()?;
        rhs.assert_vector()?;

        if self.rows != 3 || rhs.rows != 3 {
            return Err(MatrixError::VectorMustBe3DForCrossProduct);
        }

        let a = self.raw_data();
        let b = rhs.raw_data();

        Self::vector_3((
            (a[1] * b[2]) - (a[2] * b[1]),
            (a[2] * b[0]) - (a[0] * b[2]),
            (a[0] * b[1]) - (a[1] * b[0])
        ))
    }

    fn assert_vector(&self) -> Result<(), MatrixError> { 
        if self.cols != 1 {
            return Err(MatrixError::NotAVector);
        }

        Ok(())
    }
}

impl ops::Add<Matrix> for Matrix {
    type Output = Matrix;

    fn add(self, rhs: Matrix) -> Self::Output {
        let result = add(self.data, rhs.data).unwrap();

        Matrix::from_flat_vec(result, self.rows, self.cols).unwrap()
    }
}

impl ops::Sub<Matrix> for Matrix {
    type Output = Matrix;

    fn sub(self, rhs: Matrix) -> Self::Output {
        let result = sub(self.data, rhs.data).unwrap();

        Matrix::from_flat_vec(result, self.rows, self.cols).unwrap()
    }
}

impl ops::Mul<f32> for Matrix {
    type Output = Matrix;

    fn mul(self, rhs: f32) -> Self::Output {
        let result = mul(self.data, rhs).unwrap();

        Matrix::from_flat_vec(result, self.rows, self.cols).unwrap()
    }
}

pub fn add(
    lhs: Vec<f32>,
    rhs: Vec<f32>
) -> Result<Vec<f32>, MatrixError> {
    if lhs.len() != rhs.len()  {
        return Err(MatrixError::MatricesMustHaveSameDimensions);
    }

    let mut result: Vec<f32> = Vec::new();
    for (i, _) in lhs.iter().enumerate() {
        result.push(lhs[i] + rhs[i]);
    }

    Ok(result)
}

pub fn sub(
    lhs: Vec<f32>,
    rhs: Vec<f32>
) -> Result<Vec<f32>, MatrixError> {
    if lhs.len() != rhs.len() {
        return Err(MatrixError::MatricesMustHaveSameDimensions);
    }

    let mut result: Vec<f32> = Vec::new();
    for (i, _) in lhs.iter().enumerate() {
        result.push(lhs[i] - rhs[i]);
    }

    Ok(result)
}

pub fn mul(
    lhs: Vec<f32>,
    rhs: f32
) -> Result<Vec<f32>, MatrixError> {
    let mut result: Vec<f32> = Vec::new();
    for (i, _) in lhs.iter().enumerate() {
        result.push(lhs[i] * rhs);
    }

    Ok(result)
}

pub fn dot(
    lhs: (Vec<f32>, usize, usize),
    rhs: (Vec<f32>, usize, usize)
) -> Result<(Vec<f32>, usize, usize), MatrixError> {
    let (lhs, mut lhs_r, mut lhs_c) = lhs;
    let (rhs, rhs_r, rhs_c) = rhs;

    if lhs_c == rhs_c && lhs_c == 1 {
        // both sides are vectors
        lhs_c = lhs_r;
        lhs_r = 1;
    }
    
    // TODO: better algo (Solvay Strassen Algorithm)
    // https://www.baeldung.com/cs/matrix-multiplication-algorithms
    if lhs_c != rhs_r {
        return Err(MatrixError::MatrixDimensionsDoNotMatchForDotProduct);
    }

    let mut data: Vec<f32> = Vec::new();
    for row in 0..lhs_r {
        for col in 0..rhs_c {
            let mut val = 0.0;
            for k in 0..rhs_r {
                let lhs_val = get(&lhs, lhs_c, (row, k))?;
                let rhs_val = get(&rhs, rhs_c, (k, col))?;

                val += lhs_val * rhs_val;
            }

            data.push(val);
        }
    }

    Ok((data, lhs_r, rhs_c))
}

pub fn get(
    data: &Vec<f32>,
    cols: usize,
    loc: (usize, usize)
) -> Result<f32, MatrixError> {
    let (row, col) = loc;

    let index = (row * cols) + col;

    if index >= data.len() {
        return Err(MatrixError::OutOfBounds);
    }

    Ok(data[index])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_raw_data() {
        let matrix = Matrix::new(vec![
            vec![1.0, 2.0],
            vec![2.2, 7.3]
        ]).unwrap();

        assert_eq!(matrix.raw_data(), [1.0, 2.0, 2.2, 7.3]);
    }

    #[test]
    fn get_value_at_location() {
        let matrix = Matrix::new(vec![
            vec![1.0, 2.0, 2.0, 3.0],
            vec![2.2, 5.7, 9.0, 1.2],
            vec![3.6, 0.2, 7.8, 10.0]
        ]).unwrap();

        assert_eq!(matrix.get(0, 1).unwrap(), 2.0);
        assert_eq!(matrix.get(1, 2).unwrap(), 9.0);
        assert_eq!(matrix.get(1, 1).unwrap(), 5.7);
        assert_eq!(matrix.get(2, 3).unwrap(), 10.0);
    }

    #[test]
    fn get_throws_if_out_of_bounds() {
        let matrix = Matrix::new(vec![
            vec![1.0, 2.0, 2.0, 3.0],
            vec![2.2, 5.7, 9.0, 1.2],
            vec![3.6, 0.2, 7.8, 10.0]
        ]).unwrap();

        let Err(error) = matrix.get(3, 5) else {
            panic!("failed test")
        };

        assert_eq!(error, MatrixError::OutOfBounds);
    }

    #[test]
    fn should_add_matrices() {
        let m1 = Matrix::new(vec![
            vec![2.1, 3.4],
            vec![6.5, 1.2]
        ]).unwrap();
        let m2 = Matrix::new(vec![
            vec![1.2, 2.3],
            vec![5.2, 1.5]
        ]).unwrap();
        let m3 = Matrix::from_flat_vec(vec![3.3, 5.7, 11.7, 2.7], 2, 2).unwrap();

        let result = m1 + m2;

        assert_eq!(result, m3);
    }

    #[test]
    fn should_subtract_matrices() {
        let m1 = Matrix::new(vec![
            vec![2.0, 3.0],
            vec![6.0, 1.0]
        ]).unwrap();
        let m2 = Matrix::new(vec![
            vec![1.0, 2.0],
            vec![5.0, 1.0]
        ]).unwrap();
        let m3 = Matrix::from_flat_vec(vec![1.0, 1.0, 1.0, 0.0], 2, 2).unwrap();

        let result = m1 - m2;
        
        assert_eq!(result, m3);
    }

    #[test]
    fn should_multiply_by_scaler() {
        let m1 = Matrix::new(vec![
            vec![2.0, 3.0],
            vec![6.0, 1.0]
        ]).unwrap();
        let m2 = Matrix::new(vec![
            vec![4.0, 6.0],
            vec![12.0, 2.0]
        ]).unwrap();

        let result = m1 * 2.0;

        assert_eq!(result, m2);
    }

    #[test]
    fn should_perform_dot_product_on_matrices() {
        let m1 = Matrix::new(vec![
            vec![2.0, 3.0],
            vec![6.0, 1.0]
        ]).unwrap();
        let m2 = Matrix::new(vec![
            vec![3.0, 2.0, 5.0],
            vec![1.0, 3.0, 2.0]
        ]).unwrap();
        let m3 = Matrix::from_flat_vec(vec![9.0, 13.0, 16.0, 19.0, 15.0, 32.0], 2, 3).unwrap();

        let result = m1.dot(&m2).unwrap();
        
        assert_eq!(result, m3);
    }

    #[test]
    fn vector_dot_product() {
        let v1 = Matrix::vector(vec![1.0, 2.0]).unwrap();
        let v2 = Matrix::vector(vec![3.0, 4.0]).unwrap();

        assert_eq!(v1.vector_dot(&v2).unwrap(), 11.0);
    }

    #[test]
    fn vector_length() {
        let vector = Matrix::vector_3((3.0, 4.0, 5.0)).unwrap();

        assert_eq!(vector.vector_length().unwrap(), 50.0_f32.sqrt());
    }

    #[test]
    fn vector_unit() {
        let vector = Matrix::vector_3((2.0, 3.0, 4.0)).unwrap();

        assert_eq!(vector.vector_unit().unwrap(), Matrix::vector_3(
            (
                2.0 / 29.0_f32.sqrt(),
                3.0 / 29.0_f32.sqrt(),
                4.0 / 29.0_f32.sqrt()
            )
        ).unwrap());
    }

    #[test]
    fn vector_cross() {
        let v1 = Matrix::vector_3((1.0, 2.0, 3.0)).unwrap();
        let v2 = Matrix::vector_3((1.0, 5.0, 7.0)).unwrap();
        let v3 = Matrix::vector_3((-1.0, -4.0, 3.0)).unwrap();

        assert_eq!(v1.vector_cross(&v2).unwrap(), v3);
    }

    #[test]
    fn vector_mul_identity_matrix() {
        let vector = Matrix::vector_4((1.0, 2.0, 3.0, 4.0)).unwrap();
        let identity = Matrix::new(vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0]
        ]).unwrap();

        assert_eq!(identity.dot(&vector).unwrap(), vector);
    }
}
