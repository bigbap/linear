#[cfg(test)]
use super::{
    matrix::*,
    errors::*,
    functions::*
};

#[test]
fn matrix_raw_data() {
    let matrix = Matrix2::new(vec![
        1.0, 2.0,
        2.2, 7.3
    ]).unwrap();

    assert_eq!(matrix.raw_data(), [1.0, 2.0, 2.2, 7.3]);
}

#[test]
fn get_value_at_location() {
    let matrix = Matrix::<3, 4>::new(vec![
        1.0, 2.0, 2.0, 3.0,
        2.2, 5.7, 9.0, 1.2,
        3.6, 0.2, 7.8, 10.0
    ]).unwrap();

    assert_eq!(matrix.get(0, 1).unwrap(), 2.0);
    assert_eq!(matrix.get(1, 2).unwrap(), 9.0);
    assert_eq!(matrix.get(1, 1).unwrap(), 5.7);
    assert_eq!(matrix.get(2, 3).unwrap(), 10.0);
}

#[test]
fn get_throws_if_out_of_bounds() {
    let matrix = Matrix::<3, 4>::new(vec![
        1.0, 2.0, 2.0, 3.0,
        2.2, 5.7, 9.0, 1.2,
        3.6, 0.2, 7.8, 10.0
    ]).unwrap();

    let Err(error) = matrix.get(3, 5) else {
        panic!("failed test")
    };

    assert_eq!(error, MatrixError::OutOfBounds);
}

#[test]
fn matrix_transpose() {
    let matrix = Matrix::<3, 4>::new(vec![
        1.0, 2.0, 2.0, 3.0,
        2.2, 5.7, 9.0, 1.2,
        3.6, 0.2, 7.8, 10.0
    ]).unwrap();

    let transposed = Matrix::<4, 3>::new(vec![
        1.0, 2.2, 3.6,
        2.0, 5.7, 0.2,
        2.0, 9.0, 7.8,
        3.0, 1.2, 10.0
    ]).unwrap();

    assert_eq!(transpose(&matrix).unwrap(), transposed);
}

#[test]
fn vector_transpose() {
    let vector = Vector3::new(vec![
        1.0,
        2.0,
        3.0
    ]).unwrap();

    let transposed = Matrix::<1, 3>::new(vec![1.0, 2.0, 3.0]).unwrap();

    assert_eq!(transpose(&vector).unwrap(), transposed);
}

#[test]
fn matrix_addition() {
    let m1 = Matrix2::new(vec![
        2.1, 3.4,
        6.5, 1.2
    ]).unwrap();
    let m2 = Matrix2::new(vec![
        1.2, 2.3,
        5.2, 1.5
    ]).unwrap();
    let m3 = Matrix2::new(vec![3.3, 5.7, 11.7, 2.7]).unwrap();

    let result = m1 + m2;

    assert_eq!(result, m3);
}

#[test]
fn matrix_subtraction() {
    let m1 = Matrix2::new(vec![
        2.0, 3.0,
        6.0, 1.0
    ]).unwrap();
    let m2 = Matrix2::new(vec![
        1.0, 2.0,
        5.0, 1.0
    ]).unwrap();
    let m3 = Matrix2::new(vec![1.0, 1.0, 1.0, 0.0]).unwrap();

    let result = m1 - m2;

    assert_eq!(result, m3);
}

#[test]
fn matrix_multiplied_by_scaler() {
    let m1 = Matrix2::new(vec![
        2.0, 3.0,
        6.0, 1.0
    ]).unwrap();
    let m2 = Matrix2::new(vec![
        4.0, 6.0,
        12.0, 2.0
    ]).unwrap();

    let result = &m1 * 2.0;

    assert_eq!(result, m2);
}

#[test]
fn matrix_dot_product() {
    let m1 = Matrix2::new(vec![
        2.0, 3.0,
        6.0, 1.0
    ]).unwrap();
    let m2 = Matrix::<2, 3>::new(vec![
        3.0, 2.0, 5.0,
        1.0, 3.0, 2.0
    ]).unwrap();
    let m3 = Matrix::<2, 3>::new(vec![
        9.0, 13.0, 16.0,
        19.0, 15.0, 32.0
    ]).unwrap();

    let result = &m1 * &m2;

    assert_eq!(result, m3);
}

#[test]
fn vector_dot_product() {
    let v1 = Vector2::new(vec![1.0, 2.0]).unwrap();
    let v2 = Vector2::new(vec![3.0, 4.0]).unwrap();

    assert_eq!(vector_dot(&v1, &v2).unwrap(), 11.0);
}

#[test]
fn vector_length_function() {
    let vector = Vector::<3>::new(vec![3.0, 4.0, 5.0]).unwrap();

    assert_eq!(
        vector_length(&vector).unwrap(),
        50.0_f32.sqrt()
    );
}

#[test]
fn vector_unit_function() {
    let vector = Vector3::new(vec![2.0, 3.0, 4.0]).unwrap();

    assert_eq!(vector_unit(&vector).unwrap(), Vector3::new(vec![
        2.0 / 29.0_f32.sqrt(),
        3.0 / 29.0_f32.sqrt(),
        4.0 / 29.0_f32.sqrt()
    ]).unwrap());
}

#[test]
fn vector_cross_function() {
    let v1 = Vector3::new(vec![1.0, 2.0, 3.0]).unwrap();
    let v2 = Vector3::new(vec![1.0, 5.0, 7.0]).unwrap();
    let v3 = Vector3::new(vec![-1.0, -4.0, 3.0]).unwrap();

    assert_eq!(vector_cross(&v1, &v2).unwrap(), v3);
}
