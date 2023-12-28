#[cfg(test)]
use super::matrix::*;

#[test]
fn identity_matrix_4() {
    let identity = Matrix4::identity().unwrap();
    let expect = Matrix::<4, 4>::new(vec![
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    ]).unwrap();

    assert_eq!(identity, expect);
}


#[test]
fn scaling_matrix_4() {
    let vector = Vector4::new(vec![2.0, 3.0, 4.0, 5.0]).unwrap();
    let scaler = Matrix4::scaling_matrix(vector).unwrap();
    let expect = Matrix::<4, 4>::new(vec![
        2.0, 0.0, 0.0, 0.0,
        0.0, 3.0, 0.0, 0.0,
        0.0, 0.0, 4.0, 0.0,
        0.0, 0.0, 0.0, 5.0
    ]).unwrap();

    assert_eq!(scaler, expect);
}

#[test]
fn translation_matrix_4() {
    let vector = Vector4::new(vec![2.0, 3.0, 4.0, 5.0]).unwrap();
    let translation = Matrix4::translation_matrix(vector).unwrap();
    let expect = Matrix::<4, 4>::new(vec![
        1.0, 0.0, 0.0, 2.0,
        0.0, 1.0, 0.0, 3.0,
        0.0, 0.0, 1.0, 4.0,
        0.0, 0.0, 0.0, 5.0
    ]).unwrap();

    assert_eq!(translation, expect);
}

#[test]
fn identity_matrix_dot_vector() {
    let vector = Vector4::new(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let identity = Matrix4::identity().unwrap();

    let result = &identity * &vector;

    assert_eq!(result, vector);
}

#[test]
fn scaling_matrix_dot_vector() {
    let vector = Vector4::new(vec![2.0, 2.0, 0.5, 1.0]).unwrap();
    let scaling_matrix = Matrix4::scaling_matrix(vector).unwrap();
    let vector_to_transform = Vector4::new(vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    let result = &scaling_matrix * &vector_to_transform;

    assert_eq!(result, Vector4::new(vec![2.0, 4.0, 1.5, 4.0]).unwrap());
}

#[test]
fn translation_matrix_dot_vector() {
    let vector = Vector4::new(vec![2.0, 2.0, 0.5, 1.0]).unwrap();
    let translation_matrix = Matrix4::translation_matrix(vector).unwrap();
    let vector_to_translate = Vector4::new(vec![1.0, 2.0, 3.0, 1.0]).unwrap();

    let result = &translation_matrix * &vector_to_translate;

    assert_eq!(result, Vector4::new(vec![3.0, 4.0, 3.5, 1.0]).unwrap());
}
