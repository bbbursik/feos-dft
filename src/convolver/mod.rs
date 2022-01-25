use crate::geometry::{Axis, AxisGeometry, Grid};
use crate::weight_functions::{self, *};
use feos_core::EosResult;
use ndarray::prelude::*;
use ndarray::{Axis as Axis_nd, Ix1, Ix2, RemoveAxis, ScalarOperand, Slice};
use ndarray_npy::write_npy;
use num_dual::*;
use rustdct::DctNum;
use std::ops::{AddAssign, MulAssign, SubAssign};
use std::rc::Rc;

mod periodic_convolver;
mod transform;
pub use periodic_convolver::PeriodicConvolver;
use transform::*;

/// Trait for numerical convolutions for DFT.
///
/// Covers calculation of weighted densities & functional derivatives
/// from density profiles & profiles of the partial derivatives of the
/// Helmholtz energy functional.
///
/// Parametrized over data types `T` and dimension of the problem `D`.
pub trait Convolver<T, D: Dimension> {
    /// Convolve the profile with the given weight function.
    fn convolve(&self, profile: Array<T, D>, weight_function: &WeightFunction<T>) -> Array<T, D>;

    /// Calculate weighted densities via convolution from density profiles.
    fn weighted_densities(&self, density: &Array<T, D::Larger>) -> Vec<Array<T, D::Larger>>;

    /// Calculate the functional derivative via convolution from partial derivatives
    /// of the Helmholtz energy functional.
    fn functional_derivative(
        &self,
        partial_derivatives: &[Array<T, D::Larger>],
        second_partial_derivatives: &[Array<T, <D::Larger as Dimension>::Larger>],
        weighted_densities: &[Array<T, D::Larger>],
    ) -> Array<T, D::Larger>;
}

/// Base structure to hold either information about the weight function through
/// `WeightFunctionInfo` or the weight functions themselves via
/// `FFTWeightFunctions`.
#[derive(Debug, Clone)]
struct FFTWeightFunctions<T, D: Dimension> {
    /// Either number of components for simple functionals
    /// or idividual segments for group contribution methods
    pub(crate) segments: usize,
    /// Flag if local density is required in the functional
    pub(crate) local_density: bool,
    /// Container for scalar component-wise weighted densities
    pub(crate) scalar_component_weighted_densities: Vec<Array<T, D::Larger>>,
    /// Container for vector component-wise weighted densities
    pub(crate) vector_component_weighted_densities: Vec<Array<T, <D::Larger as Dimension>::Larger>>,
    /// Container for scalar FMT weighted densities
    pub(crate) scalar_fmt_weighted_densities: Vec<Array<T, D::Larger>>,
    /// Container for vector FMT weighted densities
    pub(crate) vector_fmt_weighted_densities: Vec<Array<T, <D::Larger as Dimension>::Larger>>,
}

impl<T, D: Dimension> FFTWeightFunctions<T, D> {
    /// Calculates the total number of weighted densities for each functional
    /// from multiple weight functions depending on dimension.
    pub fn n_weighted_densities(&self, dimensions: usize) -> usize {
        (if self.local_density { self.segments } else { 0 })
            + self.scalar_component_weighted_densities.len() * self.segments
            + self.vector_component_weighted_densities.len() * self.segments * dimensions
            + self.scalar_fmt_weighted_densities.len()
            + self.vector_fmt_weighted_densities.len() * dimensions
    }
}

pub struct GradConvolver<T, D: Dimension> {
    grid: Array<T, D>,
    weight_functions: Vec<WeightFunctionInfo<T>>,
    weight_constants: Vec<Array<HyperDual64, D::Larger>>,
}

// impl<T, D: Dimension + 'static> GradConvolver<T, D>
// where
//     T: DctNum + DualNum<f64> + ScalarOperand,
//     D::Larger: Dimension<Smaller = D>,
//     <D::Larger as Dimension>::Larger: Dimension<Smaller = D::Larger>,
// {
//     fn new(
//         grid: Array<T, D>,
//         weight_functions: Vec<WeightFunctionInfo<T>>, //&[WeightFunctionInfo<T>],
//     ) -> Rc<dyn Convolver<T, D>> {
//         Rc::new(Self {
//             grid,
//             weight_functions,
//         })
//     }
// }

impl<T> GradConvolver<T, Ix1>
where
    T: DctNum + DualNum<f64> + ScalarOperand + ndarray_npy::WritableElement,
{
    pub fn new(
        grid: Array<T, Ix1>,
        weight_functions: Vec<WeightFunctionInfo<T>>, //&[WeightFunctionInfo<T>],
        weight_constants: Vec<Array<HyperDual64, Ix2>>,
    ) -> Rc<dyn Convolver<T, Ix1>> {
        Rc::new(Self {
            grid,
            weight_functions,
            weight_constants,
        })
    }

    pub fn gradient(&self, f: &Array<T, Ix2>, dx: T) -> Array<T, Ix2> {
        // println!("grad version: 17:16");
        let grad = Array::from_shape_fn(f.raw_dim(), |(c, i)| {
            let width: usize = 6;
            let width_f64 = width as f64;
            let d = if i as isize - width as isize <= 0 {
                (f[(c, i + width)] - f[(c, i)]) * 0.0 // Left value --> where from?
            } else if i + width >= f.shape()[1] - 1 {
                (f[(c, i)] - f[(c, i - width)]) * 0.0
            } else {
                f[(c, i + width)] - f[(c, i - width)]
                // (f[(c, i + 1)] - f[(c, i)]) * 2.0
            };
            d / (dx * 2.0 * width_f64)
        });
        grad
    }

    // pub fn gradient(&self, f: &Array<T, Ix2>, dx: T) -> Array<T, Ix2> {
    //     // println!("grad version: 17:16");
    //     let grad = Array::from_shape_fn(f.raw_dim(), |(c, i)| {
    //         let width:usize = 1;
    //         let width_f64 = width as f64;
    //         let d = if i as isize - width as isize * 2 <= 0 {
    //             (f[(c, i + width)] - f[(c, i)]) * 2.0 // Left value --> where from?
    //         } else if i + width*2 >= f.shape()[1] - 1 {
    //             (f[(c, i)] - f[(c, i - width)]) * 2.0
    //         } else {
    //             f[(c,i-2*width)] - f[(c, i - width)] * 8.0 + f[(c, i + width)] * 8.0 - f[(c, i + 2*width)]
    //             // (f[(c, i + 1)] - f[(c, i)]) * 2.0
    //         };
    //         d / (dx * 12.0 * width_f64)
    //     });
    //     grad
    // }

    // pub fn gradient_3d(&self, f: &Array<T, Ix3>, dx: T) -> Array<T, Ix3> {
    //     // println!("grad version: 17:16");
    //     let grad = Array::from_shape_fn(f.raw_dim(), |(c1, c2, i)| {
    //         let width:usize = 2;
    //         let width_f64 = width as f64;
    //         let d = if i as isize - width as isize <= 0 {
    //             (f[(c1, c2, i + width)] - f[(c1, c2, i)]) * 2.0 // Left value --> where from?
    //         } else if i + width >= f.shape()[2] - 1 {
    //             (f[(c1, c2, i)] - f[(c1, c2, i - width)]) * 2.0
    //         } else {
    //             f[(c1, c2,i-2*width)] - f[(c1, c2, i - width)] * 8.0 + f[(c1, c2, i + width)] * 8.0 - f[(c1, c2, i + 2*width)]
    //         };
    //         d / (dx * 1z2.0 * width_f64)
    //     });
    //     grad
    // }

    // pub fn gradient_3d_new(&self, f: &Array<T, Ix3>, dx: T) -> Array<T, Ix3> {
    //     // println!("grad version: 17:16");
    //     let mut grad = Array::zeros(f.raw_dim()).into_dimensionality().unwrap();
    //     grad = f.outer_iter().map(|fi| self.gradient(fi, dx)).collect();
    //     // for (g, fi) in grad.outer_iter_mut().zip(f.outer_iter()){
    //     //     g = self.gradient(fi, dx);
    //     // }
    //     grad
    // }

    pub fn gradient_3d(&self, f: &Array<T, Ix3>, dx: T) -> Array<T, Ix3> {
        // println!("grad version: 17:16");
        let grad = Array::from_shape_fn(f.raw_dim(), |(c1, c2, i)| {
            let width: usize = 6;
            let width_f64 = width as f64;
            let d = if i as isize - width as isize <= 0 {
                (f[(c1, c2, i + width)] - f[(c1, c2, i)]) * 0.0 // Left value --> where from?
            } else if i + width >= f.shape()[2] - 1 {
                (f[(c1, c2, i)] - f[(c1, c2, i - width)]) * 0.0
            } else {
                f[(c1, c2, i + width)] - f[(c1, c2, i - width)]
            };
            d / (dx * 2.0 * width_f64)
        });
        grad
    }

    pub fn laplace(&self, f: &Array<T, Ix2>, dx: T) -> Array<T, Ix2> {
        // println!("lapl-version: 14:00");
        let lapl = Array::from_shape_fn(f.raw_dim(), |(c, i)| {
            let width: usize = 6;
            let width_f64 = width as f64;
            let d = if i as isize - width as isize <= 0 {
                // (f[(c, 2)] - f[(c, 0)]) * 0.0 // Left value --> where from?
                (f[(c, i + width * 2)] - f[(c, i + width)] * 2.0 + f[(c, i)]) * 0.0
            // Left value --> where from?
            } else if i + width >= f.shape()[1] - 1 {
                // (f[(c, f.shape()[1] - 1)] - f[(c, f.shape()[1] - 3)]) * 0.0
                (f[(c, i)] - f[(c, i - width)] * 2.0 + f[(c, i - width * 2)]) * 0.0
            } else {
                (f[(c, i + width)] - f[(c, i)] * 2.0 + f[(c, i - width)])
                // f[(c, i + 2)] - f[(c, i + 1)] * 2.0 + f[(c, i)]
            };

            d / (dx * dx) / (width_f64 * width_f64)
        });
        lapl
    }

    // pub fn laplace(&self, f: &Array<T, Ix2>, dx: T) -> Array<T, Ix2> {
    //     // println!("lapl-version: 14:00");
    //     let lapl = Array::from_shape_fn(f.raw_dim(), |(c, i)| {
    //         let d = if i == 0 {
    //             // (f[(c, 2)] - f[(c, 0)]) * 0.0 // Left value --> where from?
    //             f[(c, 6)] - f[(c, 3)] * 2.0 + f[(c, 0)] // Left value --> where from?
    //         } else if i == 1 {
    //             f[(c, 7)] - f[(c, 4)] * 2.0 + f[(c, 1)]
    //         } else if i == 2 {
    //             f[(c, 8)] - f[(c, 5)] * 2.0 + f[(c, 2)]
    //         } else if i == f.shape()[1] - 1 {
    //             // (f[(c, f.shape()[1] - 1)] - f[(c, f.shape()[1] - 3)]) * 0.0
    //             f[(c, f.shape()[1] - 1)] - f[(c, f.shape()[1] - 4)] * 2.0 + f[(c, f.shape()[1] - 7)]
    //         } else if i == f.shape()[1] - 2 {
    //             // (f[(c, f.shape()[1] - 1)] - f[(c, f.shape()[1] - 3)]) * 0.0
    //             f[(c, f.shape()[1] - 2)] - f[(c, f.shape()[1] - 5)] * 2.0 + f[(c, f.shape()[1] - 8)]
    //         } else if i == f.shape()[1] - 3 {
    //             // (f[(c, f.shape()[1] - 1)] - f[(c, f.shape()[1] - 3)]) * 0.0
    //             f[(c, f.shape()[1] - 3)] - f[(c, f.shape()[1] - 6)] * 2.0 + f[(c, f.shape()[1] - 9)]
    //         } else {
    //             (f[(c, i + 3)] - f[(c, i)] * 2.0 + f[(c, i - 3)])
    //             // f[(c, i + 2)] - f[(c, i + 1)] * 2.0 + f[(c, i)]
    //         };

    //         d / (dx * dx) / (4.0 * 4.0)
    //     });
    //     lapl
    // }
}

impl<T> Convolver<T, Ix1> for GradConvolver<T, Ix1>
where
    T: DctNum + DualNum<f64> + ScalarOperand + ndarray_npy::WritableElement, //last one just for writing into file for debugging
{
    fn convolve(
        &self,
        profile: Array<T, Ix1>,
        weight_function: &WeightFunction<T>,
    ) -> Array<T, Ix1> {
        unimplemented!();
    }

    fn weighted_densities(&self, density: &Array<T, Ix2>) -> Vec<Array<T, Ix2>> {
        println!(" Called fn weighted_densities in GradConvolver");
        let dx = self.grid[1] - self.grid[0];

        let gradient = self.gradient(density, dx);
        let laplace = self.laplace(density, dx);

        write_npy("grad_rho.npy", &gradient).unwrap();
        write_npy("lapl_rho.npy", &laplace).unwrap();

        // let temperature =
        //     HyperDual64::from(self.temperature.to_reduced(U::reference_temperature())?);

        // let weight_functions: Vec<WeightFunctionInfo<HyperDual64>> = self
        //     .dft
        //     .functional
        //     .contributions()
        //     .iter()
        //     .map(|c| c.weight_functions(temperature))
        //     .collect();

        let k0 = HyperDual64::from(0.0).derive1().derive2();
        let mut weighted_densities_vec = Vec::with_capacity(self.weight_functions.len());

        // let weight_functions_hd: Vec<WeightFunctionInfo<HyperDual64>> =
        // self.weight_functions as Vec<WeightFunctionInfo<HyperDual64>>;

        //loop over contributions
        let mut j = 0;
        for (wf, wc) in self
            .weight_functions
            .iter()
            .zip(self.weight_constants.iter())
        {
            let segments = wf.component_index.len();
            //let wf_hd = WeightFunctionInfo::from(HyperDual64::from(wf)); //brauche die wf als Hyperdual für die weight constants, aber es kommt aus Interface als f64 --> übergeben aus Interface
            // let w = wf.weight_constants(k0, 1); //can rewrite this with the corresponding function for the weight constants (i.e. scalar_weigth_const)
            let w0 = wc.mapv(|w| w.re);
            let w1 = wc.mapv(|w| -w.eps1[0]);
            let w2 = wc.mapv(|w| -0.5 * w.eps1eps2[(0, 0)]);

            // number of weighted densities
            let n_wd = wf.n_weighted_densities(1);

            // Allocating new array for intended weighted densities
            let mut dim = vec![n_wd];
            density.shape().iter().skip(1).for_each(|&d| dim.push(d));

            let mut weighted_densities = Array::zeros(dim).into_dimensionality().unwrap();

            // Initilaizing row index for non-local weighted densities
            let mut k = 0;

            // Assigning possible local densities to the front of the array
            if wf.local_density {
                weighted_densities
                    .slice_axis_mut(Axis_nd(0), Slice::from(0..segments))
                    .assign(&density);
                k += segments;
            }

            // Calculating weighted densities {scalar, component}
            for wf_i in &wf.scalar_component_weighted_densities {
                for (i, ((rho, lapl), mut res)) in density
                    .outer_iter()
                    .zip(laplace.outer_iter())
                    .zip(
                        weighted_densities
                            .slice_axis_mut(Axis_nd(0), Slice::from(k..k + segments))
                            .outer_iter_mut(),
                    )
                    .enumerate()
                {
                    res.assign(
                        &(&rho * w0.slice(s![k..k + segments, ..]).into_diag()[i]
                            + &lapl * w2.slice(s![k..k + segments, ..]).into_diag()[i]),
                    );
                }
                k += segments;
            }

            // Calculating weighted densities {vector, component}
            // !! WORKS FOR 1D ONLY !!
            for wf_i in &wf.vector_component_weighted_densities {
                for (i, (grad, mut res)) in gradient
                    .outer_iter()
                    .zip(
                        weighted_densities
                            .slice_axis_mut(Axis_nd(0), Slice::from(k..k + segments))
                            .outer_iter_mut(),
                    )
                    .enumerate()
                {
                    res.assign(&(&grad * w1.slice(s![k..k + segments, ..]).into_diag()[i]));
                }
                k += segments;
            }

            // Calculating weighted densities {scalar, FMT}
            for wf_i in &wf.scalar_fmt_weighted_densities {
                for (i, (rho, lapl)) in density.outer_iter().zip(laplace.outer_iter()).enumerate() {
                    weighted_densities.index_axis_mut(Axis_nd(0), k).add_assign(
                        &(&rho * w0.slice(s![k, ..])[i] + &lapl * w2.slice(s![k, ..])[i]),
                    );
                }
                k += 1;
            }

            // Calculating weighted densities {vector, FMT}
            // !! WORKS FOR 1D ONLY I think!!
            for wf_i in &wf.vector_fmt_weighted_densities {
                for (i, grad) in gradient.outer_iter().enumerate() {
                    weighted_densities
                        .index_axis_mut(Axis_nd(0), k)
                        .add_assign(&(&grad * w1.slice(s![k, ..])[i]));
                }
                k += 1;
            }

            let filename3 = j.to_string().to_owned();
            write_npy(filename3 + "_wd.npy", &weighted_densities).unwrap();
            j = j + 1;

            weighted_densities_vec.push(weighted_densities);
        }

        weighted_densities_vec
    }

    fn functional_derivative(
        &self,
        partial_derivatives: &[Array<T, Ix2>],
        second_partial_derivatives: &[Array<T, Ix3>], //only for 2nd variant of functional derivative
        weighted_densities: &[Array<T, Ix2>],
    ) -> Array<T, Ix2> {
        // println!(" Called fn functional_derivative in GradConvolver");

        // let temperature = self.temperature.to_reduced(U::reference_temperature())?;
        // println!("Version of 8:50");
        let mut dim = vec![(self.weight_functions[0]).component_index.len()];
        partial_derivatives[0]
            .shape()
            .iter()
            .skip(1)
            .for_each(|&d| dim.push(d));
        // let mut functional_deriv = Array::zeros(dim).into_dimensionality().unwrap();

        // let densities = self.density.to_reduced(U::reference_density())?; //.view()
        let dx = self.grid[1] - self.grid[0];

        let k0 = HyperDual64::from(0.0).derive1().derive2();

        // let weighted_densities = self.weighted_densities()?;
        // let contributions = self.dft.functional.contributions();
        let mut functional_derivative_0: Array<T, Ix2> =
            Array::zeros(dim).into_dimensionality().unwrap();
        let mut functional_derivative_1: Array<T, Ix2> =
            Array::zeros(functional_derivative_0.raw_dim())
                .into_dimensionality()
                .unwrap();
        let mut functional_derivative_2: Array<T, Ix2> =
            Array::zeros(functional_derivative_0.raw_dim())
                .into_dimensionality()
                .unwrap();
        for (i, ((wf, wc), pd)) in self
            .weight_functions
            .iter()
            .zip(self.weight_constants.iter())
            .zip(partial_derivatives.iter())
            .enumerate()
        {
            // let wf = c.weight_functions(HyperDual64::from(temperature));
            // let w = wf.weight_constants(k0, 1);
            let w0 = wc.mapv(|w| w.re);
            let w1 = wc.mapv(|w| -w.eps1[0]);
            let w2 = wc.mapv(|w| -0.5 * w.eps1eps2[(0, 0)]);

            let segments = wf.component_index.len();
            // let nwd = wd.shape()[0];
            // let ngrid = wd.len() / nwd;
            // let mut phi = Array::zeros(densities.raw_dim().remove_axis(Axis_nd(0)));
            // let mut first_partial_derivative = Array::zeros(wd.raw_dim());
            // //let mut spd = Array::zeros(wd.raw_dim());
            // c.first_partial_derivatives(
            //     temperature,
            //     wd.into_shape((nwd, ngrid)).unwrap(),
            //     phi.view_mut().into_shape(ngrid).unwrap(),
            //     first_partial_derivative
            //         .view_mut()
            //         .into_shape((nwd, ngrid))
            //         .unwrap(),
            // )?;

            // calculate gradients of partial derivatives
            // !! MAKES SENSE ONLY IN 1D FOR NOW!!

            let grad_first_partial_derivative = self.gradient(pd, dx);
            let lapl_first_partial_derivative = self.laplace(pd, dx);
            let filename0 = i.to_string().to_owned();
            write_npy(filename0 + "_grad_fpd.npy", &grad_first_partial_derivative).unwrap();
            let filename1 = i.to_string().to_owned();
            write_npy(filename1 + "_lapl_fpd.npy", &lapl_first_partial_derivative).unwrap();

            let filename2 = i.to_string().to_owned();
            write_npy(filename2 + "_firstpd.npy", pd).unwrap();
            // let filename3 = i.to_string().to_owned();
            // write_npy(filename3 + "_wd.npy", pd).unwrap();
            // println!("write npy");

            // Initilaizing row index for non-local functional derivative
            let mut k = 0;

            // Assigning possible local densities to the front of the array
            if wf.local_density {
                functional_derivative_0 += &pd.slice_axis(Axis_nd(0), Slice::from(..segments));
                k += segments;
            }

            // Calculating functional derivative {scalar, component}
            for wf_i in &wf.scalar_component_weighted_densities {
                for (i, (((fpd, lapl), mut res0), mut res2)) in pd
                    .slice_axis(Axis_nd(0), Slice::from(k..k + segments))
                    .outer_iter()
                    .zip(
                        lapl_first_partial_derivative
                            .slice_axis(Axis_nd(0), Slice::from(k..k + segments))
                            .outer_iter(),
                    )
                    .zip(functional_derivative_0.outer_iter_mut())
                    .zip(functional_derivative_2.outer_iter_mut())
                    .enumerate()
                {
                    //res.add_assign(
                    //    &(&fpd * w0.slice(s![k..k + segments, ..]).into_diag()[i]
                    //        + &lapl * w2.slice(s![k..k + segments, ..]).into_diag()[i]),
                    //);
                    res0.add_assign(&(&fpd * w0.slice(s![k..k + segments, ..]).into_diag()[i]));
                    res2.add_assign(&(&lapl * w2.slice(s![k..k + segments, ..]).into_diag()[i]));
                }
                k += segments;
            }

            // Calculating functional derivative {vector, component}
            for wf_i in &wf.vector_component_weighted_densities {
                for (i, (grad, mut res1)) in grad_first_partial_derivative
                    .slice_axis(Axis_nd(0), Slice::from(k..k + segments))
                    .outer_iter()
                    .zip(functional_derivative_1.outer_iter_mut())
                    .enumerate()
                {
                    res1.add_assign(&(&grad * (-w1.slice(s![k..k + segments, ..]).into_diag()[i])));
                }
                k += segments;
            }

            // Calculating functional derivative {scalar, FMT}
            for wf_i in &wf.scalar_fmt_weighted_densities {
                for (i, (mut res0, mut res2)) in functional_derivative_0
                    .outer_iter_mut()
                    .zip(functional_derivative_2.outer_iter_mut())
                    .enumerate()
                {
                    /* res.add_assign(
                        &(&first_partial_derivative.index_axis(Axis_nd(0), k)
                            * w0.slice(s![k, ..])[i]
                            + &lapl_first_partial_derivative.index_axis(Axis_nd(0), k)
                                * w2.slice(s![k, ..])[i]),
                    );*/
                    res0.add_assign(&(&pd.index_axis(Axis_nd(0), k) * w0.slice(s![k, ..])[i]));
                    res2.add_assign(
                        &(&lapl_first_partial_derivative.index_axis(Axis_nd(0), k)
                            * w2.slice(s![k, ..])[i]),
                    );
                }
                k += 1;
            }

            // Calculating functional derivative {vector, FMT}
            // !! WORKS FOR 1D ONLY!!
            for wf_i in &wf.vector_fmt_weighted_densities {
                for (i, mut res1) in functional_derivative_1.outer_iter_mut().enumerate() {
                    res1.add_assign(
                        &(&grad_first_partial_derivative.index_axis(Axis_nd(0), k)
                            * (-w1.slice(s![k, ..])[i])),
                    );
                }
                k += 1;
            }
        }

        // Ok(vec![
        //     functional_derivative_0,
        //     functional_derivative_1,
        //     functional_derivative_2,
        // ])

        write_npy("fd0.npy", &functional_derivative_0).unwrap();
        write_npy("fd1.npy", &functional_derivative_1).unwrap();
        write_npy("fd2.npy", &functional_derivative_2).unwrap();

        let result = functional_derivative_0 + functional_derivative_1 + functional_derivative_2;
        result
    }

    /*
    //  This is v2 of the functional derivative, where gradisents are calculated for individual terms
    fn functional_derivative(
        &self,
        partial_derivatives: &[Array<T, Ix2>],
        second_partial_derivatives: &[Array<T, Ix3>],
        weighted_densities: &[Array<T, Ix2>],
    ) -> Array<T, Ix2> {
        // let temperature = self.temperature.to_reduced(U::reference_temperature())?;
        // // println!("Version of 12:01");
        // let densities = self.density.to_reduced(U::reference_density())?; //.view()

        let mut dim = vec![(self.weight_functions[0]).component_index.len()];
        partial_derivatives[0]
            .shape()
            .iter()
            .skip(1)
            .for_each(|&d| dim.push(d));

        let dx = self.grid[1] - self.grid[0];

        let k0 = HyperDual64::from(0.0).derive1().derive2();

        // let weighted_densities = self.weighted_densities()?;
        // let contributions = self.dft.functional.contributions();
        let mut functional_derivative_0: Array<T, Ix2> =
            Array::zeros(dim).into_dimensionality().unwrap();
        let mut functional_derivative_1: Array<T, Ix2> =
            Array::zeros(functional_derivative_0.raw_dim())
                .into_dimensionality()
                .unwrap();
        let mut functional_derivative_2: Array<T, Ix2> =
            Array::zeros(functional_derivative_0.raw_dim())
                .into_dimensionality()
                .unwrap();

        // let weighted_densities = self.local_weighted_densities()?;
        // let contributions = self.dft.functional.contributions();

        for (i, ((((wf, wc), pd), secparder), wd)) in self
        // for (i, (((wf, wc), pd), secparder)) in self
            .weight_functions
            .iter()
            .zip(self.weight_constants.iter())
            .zip(partial_derivatives.iter())
            .zip(second_partial_derivatives.iter())
            .zip(weighted_densities.iter())
            .enumerate()
        {
            let segments = wf.component_index.len();
            //let wf_hd = WeightFunctionInfo::from(HyperDual64::from(wf)); //brauche die wf als Hyperdual für die weight constants, aber es kommt aus Interface als f64 --> übergeben aus Interface
            // let w = wf.weight_constants(k0, 1); //can rewrite this with the corresponding function for the weight constants (i.e. scalar_weigth_const)
            let w0 = wc.mapv(|w| w.re);
            let w1 = wc.mapv(|w| -w.eps1[0]);
            let w2 = wc.mapv(|w| -0.5 * w.eps1eps2[(0, 0)]);

            // let nwd = wd.shape()[0];
            // let ngrid = wd.len() / nwd;
            // let mut dim = vec![nwd, nwd];
            // wd.shape().iter().skip(1).for_each(|&d| dim.push(d));

            // let mut phi = Array::zeros(densities.raw_dim().remove_axis(Axis_nd(0)));
            // let mut first_partial_derivative = Array::zeros(wd.raw_dim());
            // let mut second_partial_derivative: Array<_, Ix3> =
            // Array::zeros(dim).into_dimensionality().unwrap();
            //let mut spd = Array::zeros(wd.raw_dim());
            let grad_weighted_density = self.gradient(wd, dx);
            let lapl_weighted_density = self.laplace(wd, dx);

            // c.second_partial_derivatives(
            //     temperature,
            //     wd.into_shape((nwd, ngrid)).unwrap(),
            //     phi.view_mut().into_shape(ngrid).unwrap(),
            //     first_partial_derivative
            //         .view_mut()
            //         .into_shape((nwd, ngrid))
            //         .unwrap(),
            //     second_partial_derivative
            //         .view_mut()
            //         .into_shape((nwd, nwd, ngrid))
            //         .unwrap(),
            // )?;

            // calculate gradients of partial derivatives
            let grad_first_partial_derivative = self.gradient(pd, dx);
            let lapl_first_partial_derivative = self.laplace(pd, dx);
            let mut grad_second_partial_derivative= self.gradient_3d(secparder,dx);
            // let mut grad_second_partial_derivative: Array3<T> = Array::zeros(secparder.raw_dim());
            // let mut grad_second_partial_derivative: Array3<f64> =
            //     Array::zeros(second_partial_derivative.raw_dim());

            // for (spd, mut res) in secparder
            //     .outer_iter()
            //     .zip(grad_second_partial_derivative.outer_iter_mut())
            // {
            //     let grad = self.gradient(spd, dx);
            //     res.assign(&grad);
            //     // res.assign(&self.gradient(spd, dx));
            // }

            // grad_second_partial_derivative.assign(
            //     &second_partial_derivative
            //         .outer_iter()
            //         .map(|x| self.gradient(&x.to_owned(), dx))
            //         .collect(),
            // );

            // grad_second_partial_derivative
            // .outer_iter_mut()
            // .for_each(|x| self.gradient(&x, dx)?);

            // Initilaizing row index for non-local functional derivative
            let mut k = 0;

            // Assigning possible local densities to the front of the array
            if wf.local_density {
                functional_derivative_0 += &pd.slice_axis(Axis_nd(0), Slice::from(..segments));
                k += segments;
            }

            // Calculating functional derivative {scalar, component}
            for wf_i in &wf.scalar_component_weighted_densities {
                for (
                    i,
                    ((fpd, spds), gradients_spd), // mut res0), //, mut res2), //((((((fpd, spds), gradients_spd), grad_wd), lapl_wd), mut res0), mut res2),
                ) in pd
                    .slice_axis(Axis_nd(0), Slice::from(k..k + segments))
                    .outer_iter()
                    .zip(
                        secparder
                            .slice_axis(Axis_nd(0), Slice::from(k..k + segments))
                            .outer_iter(),
                    )
                    .zip(
                        grad_second_partial_derivative
                            .slice_axis(Axis_nd(0), Slice::from(k..k + segments))
                            .outer_iter(),
                    )
                    //.zip(functional_derivative_0.outer_iter_mut())
                    //.zip(functional_derivative_2.outer_iter_mut())
                    .enumerate()
                {
                    functional_derivative_0
                        .index_axis_mut(Axis_nd(0), i)
                        .add_assign(&(&fpd * w0.slice(s![k..k + segments, ..]).into_diag()[i]));

                    for (((spd, grad_spd), grad_wd), lapl_wd) in spds
                        .outer_iter()
                        .zip(gradients_spd.outer_iter())
                        .zip(grad_weighted_density.outer_iter())
                        .zip(lapl_weighted_density.outer_iter())
                    {
                        functional_derivative_2
                            .index_axis_mut(Axis_nd(0), i)
                            .add_assign(
                                &((&grad_spd * &grad_wd + &spd * &lapl_wd)
                                    * w2.slice(s![k..k + segments, ..]).into_diag()[i]),
                            );
                    }
                    //res.add_assign(
                    //    &(&fpd * w0.slice(s![k..k + segments, ..]).into_diag()[i]
                    //        + &lapl * w2.slice(s![k..k + segments, ..]).into_diag()[i]),
                    //);
                }
                k += segments;
            }

            // Calculating functional derivative {vector, component}
            for wf_i in &wf.vector_component_weighted_densities {
                for (i, (spds, mut res1)) in secparder
                    .slice_axis(Axis_nd(0), Slice::from(k..k + segments))
                    .outer_iter()
                    .zip(functional_derivative_1.outer_iter_mut())
                    .enumerate()
                {
                    for (spd, grad_wd) in spds.outer_iter().zip(grad_weighted_density.outer_iter())
                    {
                        res1.add_assign(
                            &(&spd * (-1.0)
                                * &grad_wd
                                * (w1.slice(s![k..k + segments, ..]).into_diag()[i])),
                        );
                    }
                }
                k += segments;
            }

            // Calculating functional derivative {scalar, FMT}
            for wf_i in &wf.scalar_fmt_weighted_densities {
                for (i, (mut res0, mut res2)) in functional_derivative_0
                    .outer_iter_mut()
                    .zip(functional_derivative_2.outer_iter_mut())
                    .enumerate()
                {
                    res0.add_assign(&(&pd.index_axis(Axis_nd(0), k) * w0.slice(s![k, ..])[i]));

                    for (((spd, grad_spd), grad_wd), lapl_wd) in secparder
                        .index_axis(Axis_nd(0), k)
                        .outer_iter()
                        .zip(
                            grad_second_partial_derivative
                                .index_axis(Axis_nd(0), k)
                                .outer_iter(),
                        )
                        .zip(grad_weighted_density.outer_iter())
                        .zip(lapl_weighted_density.outer_iter())
                    {
                        res2.add_assign(
                            &((&grad_spd * &grad_wd + &spd * &lapl_wd) * w2.slice(s![k, ..])[i]),
                        );
                    }
                    /* res.add_assign(
                        &(&first_partial_derivative.index_axis(Axis_nd(0), k)
                            * w0.slice(s![k, ..])[i]
                            + &lapl_first_partial_derivative.index_axis(Axis_nd(0), k)
                                * w2.slice(s![k, ..])[i]),
                    );*/
                }
                k += 1;
            }

            // Calculating functional derivative {vector, FMT}
            // !! WORKS FOR 1D ONLY!!
            for wf_i in &wf.vector_fmt_weighted_densities {
                for (i, mut res1) in functional_derivative_1.outer_iter_mut().enumerate() {
                    for (spd, grad_wd) in secparder
                        .index_axis(Axis_nd(0), k)
                        .outer_iter()
                        .zip(grad_weighted_density.outer_iter())
                    {
                        res1.add_assign(&(&spd *(-1.0)* &grad_wd * w1.slice(s![k, ..])[i]));
                    }
                }
                k += 1;
            }
        }

        // Ok(vec![
        //     functional_derivative_0,
        //     functional_derivative_1,
        //     functional_derivative_2,
        // ])

        // let filename0 = i.to_string().to_owned();
        write_npy( "fd0.npy", &functional_derivative_0).unwrap();
        write_npy( "fd1.npy", &functional_derivative_1).unwrap();
        write_npy( "fd2.npy", &functional_derivative_2).unwrap();


        let result = functional_derivative_0 + functional_derivative_1 + functional_derivative_2;
        result
    } */
}

/// Convolver for 1-D, 2-D & 3-D systems using FFT algorithms to efficiently
/// compute convolutions in Fourier space.
///
/// Parametrized over the data type `T` and the dimension `D`.
#[derive(Clone)]
pub struct ConvolverFFT<T, D: Dimension> {
    /// k vectors
    k_abs: Array<f64, D>,
    /// Vector of weight functions for each component in multiple dimensions.
    weight_functions: Vec<FFTWeightFunctions<T, D>>,
    /// Lanczos sigma factor
    lanczos_sigma: Option<Array<f64, D>>,
    /// Possibly curvilinear Fourier transform in the first dimension
    transform: Rc<dyn FourierTransform<T>>,
    /// Vector of additional cartesian Fourier transforms in the other dimensions
    cartesian_transforms: Vec<Rc<CartesianTransform<T>>>,
}

impl<T, D: Dimension + RemoveAxis + 'static> ConvolverFFT<T, D>
where
    T: DctNum + DualNum<f64> + ScalarOperand,
    D::Larger: Dimension<Smaller = D>,
    D::Smaller: Dimension<Larger = D>,
    <D::Larger as Dimension>::Larger: Dimension<Smaller = D::Larger>,
{
    /// Create the appropriate FFT convolver for the given grid.
    pub fn plan(
        grid: &Grid,
        weight_functions: &[WeightFunctionInfo<T>],
        lanczos: Option<i32>,
    ) -> Rc<dyn Convolver<T, D>> {
        match grid {
            Grid::Polar(r) => CurvilinearConvolver::new(r, &[], weight_functions, lanczos),
            Grid::Spherical(r) => CurvilinearConvolver::new(r, &[], weight_functions, lanczos),
            Grid::Cartesian1(z) => Self::new(Some(z), &[], weight_functions, lanczos),
            Grid::Cylindrical { r, z } => {
                CurvilinearConvolver::new(r, &[z], weight_functions, lanczos)
            }
            Grid::Cartesian2(x, y) => Self::new(Some(x), &[y], weight_functions, lanczos),
            Grid::Periodical2(x, y) => PeriodicConvolver::new(&[x, y], weight_functions, lanczos),
            Grid::Cartesian3(x, y, z) => Self::new(Some(x), &[y, z], weight_functions, lanczos),
            Grid::Periodical3(x, y, z) => {
                PeriodicConvolver::new(&[x, y, z], weight_functions, lanczos)
            }
        }
    }
}

impl<T, D: Dimension + 'static> ConvolverFFT<T, D>
where
    T: DctNum + DualNum<f64> + ScalarOperand,
    D::Larger: Dimension<Smaller = D>,
    <D::Larger as Dimension>::Larger: Dimension<Smaller = D::Larger>,
{
    fn new(
        axis: Option<&Axis>,
        cartesian_axes: &[&Axis],
        weight_functions: &[WeightFunctionInfo<T>],
        lanczos: Option<i32>,
    ) -> Rc<dyn Convolver<T, D>> {
        // initialize the Fourier transform
        let mut cartesian_transforms = Vec::with_capacity(cartesian_axes.len());
        let mut k_vec = Vec::with_capacity(cartesian_axes.len() + 1);
        let mut lengths = Vec::with_capacity(cartesian_axes.len() + 1);
        let (transform, k_x) = match axis {
            Some(axis) => match axis.geometry {
                AxisGeometry::Cartesian => CartesianTransform::new(axis),
                AxisGeometry::Polar => PolarTransform::new(axis),
                AxisGeometry::Spherical => SphericalTransform::new(axis),
            },
            None => NoTransform::new(),
        };
        k_vec.push(k_x);
        lengths.push(axis.map_or(1.0, |axis| axis.length()));
        for ax in cartesian_axes {
            let (transform, k_x) = CartesianTransform::new_cartesian(ax);
            cartesian_transforms.push(transform);
            k_vec.push(k_x);
            lengths.push(ax.length());
        }

        // Calculate the full k vectors
        let mut dim = vec![k_vec.len()];
        k_vec.iter().for_each(|k_x| dim.push(k_x.len()));
        let mut k: Array<_, D::Larger> = Array::zeros(dim).into_dimensionality().unwrap();
        let mut k_abs = Array::zeros(k.raw_dim().remove_axis(Axis(0)));
        for (i, (mut k_i, k_x)) in k.outer_iter_mut().zip(k_vec.iter()).enumerate() {
            k_i.lanes_mut(Axis_nd(i))
                .into_iter()
                .for_each(|mut l| l.assign(k_x));
            k_abs.add_assign(&k_i.mapv(|k| k.powi(2)));
        }
        k_abs.map_inplace(|k| *k = k.sqrt());

        // Lanczos sigma factor
        let lanczos_sigma = lanczos.map(|exp| {
            let mut lanczos = Array::ones(k_abs.raw_dim());
            for (i, (k_x, &l)) in k_vec.iter().zip(lengths.iter()).enumerate() {
                let points = k_x.len();
                let m2 = if points % 2 == 0 {
                    points as f64 + 2.0
                } else {
                    points as f64 + 1.0
                };
                let l_x = k_x.mapv(|k| (k * l / m2).sph_j0().powi(exp));
                for mut l in lanczos.lanes_mut(Axis_nd(i)) {
                    l.mul_assign(&l_x);
                }
            }
            lanczos
        });

        // calculate weight functions in Fourier space and weight constants
        let mut fft_weight_functions = Vec::with_capacity(weight_functions.len());
        for wf in weight_functions {
            // Calculates the weight functions values from `k_abs`
            // Pre-allocation of empty `Vec`
            let mut scal_comp = Vec::with_capacity(wf.scalar_component_weighted_densities.len());
            // Filling array with scalar component-wise weight functions
            for wf_i in &wf.scalar_component_weighted_densities {
                scal_comp.push(wf_i.fft_scalar_weight_functions(&k_abs, &lanczos_sigma));
            }

            // Pre-allocation of empty `Vec`
            let mut vec_comp = Vec::with_capacity(wf.vector_component_weighted_densities.len());
            // Filling array with vector-valued component-wise weight functions
            for wf_i in &wf.vector_component_weighted_densities {
                vec_comp.push(wf_i.fft_vector_weight_functions(&k_abs, &k, &lanczos_sigma));
            }

            // Pre-allocation of empty `Vec`
            let mut scal_fmt = Vec::with_capacity(wf.scalar_fmt_weighted_densities.len());
            // Filling array with scalar FMT weight functions
            for wf_i in &wf.scalar_fmt_weighted_densities {
                scal_fmt.push(wf_i.fft_scalar_weight_functions(&k_abs, &lanczos_sigma));
            }

            // Pre-allocation of empty `Vec`
            let mut vec_fmt = Vec::with_capacity(wf.vector_fmt_weighted_densities.len());
            // Filling array with vector-valued FMT weight functions
            for wf_i in &wf.vector_fmt_weighted_densities {
                vec_fmt.push(wf_i.fft_vector_weight_functions(&k_abs, &k, &lanczos_sigma));
            }

            // Initializing `FFTWeightFunctions` structure
            fft_weight_functions.push(FFTWeightFunctions::<_, D> {
                segments: wf.component_index.len(),
                local_density: wf.local_density,
                scalar_component_weighted_densities: scal_comp,
                vector_component_weighted_densities: vec_comp,
                scalar_fmt_weighted_densities: scal_fmt,
                vector_fmt_weighted_densities: vec_fmt,
            });
        }

        // Return `FFTConvolver<T, D>`
        Rc::new(Self {
            k_abs,
            weight_functions: fft_weight_functions,
            lanczos_sigma,
            transform,
            cartesian_transforms,
        })
    }
}

impl<T, D: Dimension> ConvolverFFT<T, D>
where
    T: DctNum + DualNum<f64> + ScalarOperand,
    D::Larger: Dimension<Smaller = D>,
    <D::Larger as Dimension>::Larger: Dimension<Smaller = D::Larger>,
{
    fn forward_transform(&self, f: ArrayView<T, D>, vector_index: Option<usize>) -> Array<T, D> {
        let mut dim = vec![self.k_abs.shape()[0]];
        f.shape().iter().skip(1).for_each(|&d| dim.push(d));
        let mut result: Array<_, D> = Array::zeros(dim.clone()).into_dimensionality().unwrap();
        for (f, r) in f
            .lanes(Axis_nd(0))
            .into_iter()
            .zip(result.lanes_mut(Axis_nd(0)))
        {
            self.transform
                .forward_transform(f, r, vector_index.map_or(true, |ind| ind != 0));
        }
        for (i, transform) in self.cartesian_transforms.iter().enumerate() {
            dim[i + 1] = self.k_abs.shape()[i + 1];
            let mut res: Array<_, D> = Array::zeros(dim.clone()).into_dimensionality().unwrap();
            for (f, r) in result
                .lanes(Axis_nd(i + 1))
                .into_iter()
                .zip(res.lanes_mut(Axis_nd(i + 1)))
            {
                transform.forward_transform(f, r, vector_index.map_or(true, |ind| ind != i + 1));
            }
            result = res;
        }

        result
    }

    fn forward_transform_comps(
        &self,
        f: ArrayView<T, D::Larger>,
        vector_index: Option<usize>,
    ) -> Array<T, D::Larger> {
        let mut dim = vec![f.shape()[0]];
        self.k_abs.shape().iter().for_each(|&d| dim.push(d));
        let mut result = Array::zeros(dim).into_dimensionality().unwrap();
        for (f, mut r) in f.outer_iter().zip(result.outer_iter_mut()) {
            r.assign(&self.forward_transform(f, vector_index));
        }
        result
    }

    fn back_transform(
        &self,
        mut f: ArrayViewMut<T, D>,
        mut result: ArrayViewMut<T, D>,
        vector_index: Option<usize>,
    ) {
        let mut dim = vec![result.shape()[0]];
        f.shape().iter().skip(1).for_each(|&d| dim.push(d));
        let mut res: Array<_, D> = Array::zeros(dim.clone()).into_dimensionality().unwrap();
        for (f, r) in f
            .lanes_mut(Axis_nd(0))
            .into_iter()
            .zip(res.lanes_mut(Axis_nd(0)))
        {
            self.transform
                .back_transform(f, r, vector_index.map_or(true, |ind| ind != 0));
        }
        for (i, transform) in self.cartesian_transforms.iter().enumerate() {
            dim[i + 1] = result.shape()[i + 1];
            let mut res2: Array<_, D> = Array::zeros(dim.clone()).into_dimensionality().unwrap();
            for (f, r) in res
                .lanes_mut(Axis_nd(i + 1))
                .into_iter()
                .zip(res2.lanes_mut(Axis_nd(i + 1)))
            {
                transform.back_transform(f, r, vector_index.map_or(true, |ind| ind != i + 1));
            }
            res = res2;
        }

        result.assign(&res);
    }

    fn back_transform_comps(
        &self,
        mut f: Array<T, D::Larger>,
        mut result: ArrayViewMut<T, D::Larger>,
        vector_index: Option<usize>,
    ) {
        for (f, r) in f.outer_iter_mut().zip(result.outer_iter_mut()) {
            self.back_transform(f, r, vector_index);
        }
    }
}

impl<T, D: Dimension> Convolver<T, D> for ConvolverFFT<T, D>
where
    T: DctNum + ScalarOperand + DualNum<f64>,
    D::Larger: Dimension<Smaller = D>,
    <D::Larger as Dimension>::Larger: Dimension<Smaller = D::Larger>,
{
    fn convolve(&self, profile: Array<T, D>, weight_function: &WeightFunction<T>) -> Array<T, D> {
        // Forward transform
        let f_k = self.forward_transform(profile.view(), None);

        // calculate weight function
        let w = weight_function
            .fft_scalar_weight_functions(&self.k_abs, &self.lanczos_sigma)
            .index_axis_move(Axis(0), 0);

        // Inverse transform
        let mut result = Array::zeros(profile.raw_dim());
        self.back_transform((f_k * w).view_mut(), result.view_mut(), None);
        result
    }

    fn weighted_densities(&self, density: &Array<T, D::Larger>) -> Vec<Array<T, D::Larger>> {
        println!("called fn weighted densities in FFT convolver");
        // Applying FFT to each row of the matrix `rho` saving the result in `rho_k`
        let rho_k = self.forward_transform_comps(density.view(), None);

        // Iterate over all contributions
        let mut weighted_densities_vec = Vec::with_capacity(self.weight_functions.len());
        for wf in &self.weight_functions {
            // number of weighted densities
            let n_wd = wf.n_weighted_densities(density.ndim() - 1);

            // Allocating new array for intended weighted densities
            let mut dim = vec![n_wd];
            density.shape().iter().skip(1).for_each(|&d| dim.push(d));
            let mut weighted_densities = Array::zeros(dim).into_dimensionality().unwrap();

            // Initilaizing row index for non-local weighted densities
            let mut k = 0;

            // Assigning possible local densities to the front of the array
            if wf.local_density {
                weighted_densities
                    .slice_axis_mut(Axis(0), Slice::from(0..wf.segments))
                    .assign(density);
                k += wf.segments;
            }

            // Calculating weighted densities {scalar, component}
            for wf_i in &wf.scalar_component_weighted_densities {
                self.back_transform_comps(
                    &rho_k * wf_i,
                    weighted_densities.slice_axis_mut(Axis(0), Slice::from(k..k + wf.segments)),
                    None,
                );
                k += wf.segments;
            }

            // Calculating weighted densities {vector, component}
            for wf_i in &wf.vector_component_weighted_densities {
                for (i, wf_i) in wf_i.outer_iter().enumerate() {
                    self.back_transform_comps(
                        &rho_k * &wf_i,
                        weighted_densities.slice_axis_mut(Axis(0), Slice::from(k..k + wf.segments)),
                        Some(i),
                    );
                    k += wf.segments;
                }
            }

            // Calculating weighted densities {scalar, FMT}
            for wf_i in &wf.scalar_fmt_weighted_densities {
                self.back_transform(
                    (&rho_k * wf_i).sum_axis(Axis(0)).view_mut(),
                    weighted_densities.index_axis_mut(Axis(0), k),
                    None,
                );
                k += 1;
            }

            // Calculating weighted densities {vector, FMT}
            for wf_i in &wf.vector_fmt_weighted_densities {
                for (i, wf_i) in wf_i.outer_iter().enumerate() {
                    self.back_transform(
                        (&rho_k * &wf_i).sum_axis(Axis(0)).view_mut(),
                        weighted_densities.index_axis_mut(Axis(0), k),
                        Some(i),
                    );
                    k += 1;
                }
            }

            // add weighted densities for this contribution to the result
            weighted_densities_vec.push(weighted_densities);
        }
        // Return
        weighted_densities_vec
    }

    fn functional_derivative(
        &self,
        partial_derivatives: &[Array<T, D::Larger>],
        second_partial_derivatives: &[Array<T, <D::Larger as Dimension>::Larger>],
        weighted_densities: &[Array<T, D::Larger>],
    ) -> Array<T, D::Larger> {
        // Allocate arrays for the result, the local contribution to the functional derivative,
        // the functional derivative in Fourier space, and the bulk contributions
        let mut dim = vec![self.weight_functions[0].segments];
        partial_derivatives[0]
            .shape()
            .iter()
            .skip(1)
            .for_each(|&d| dim.push(d));
        let mut functional_deriv = Array::zeros(dim).into_dimensionality().unwrap();
        let mut functional_deriv_local = Array::zeros(functional_deriv.raw_dim());
        let mut dim = vec![self.weight_functions[0].segments];
        self.k_abs.shape().iter().for_each(|&d| dim.push(d));
        let mut functional_deriv_k = Array::zeros(dim).into_dimensionality().unwrap();

        // Iterate over all contributions
        for (pd, wf) in partial_derivatives.iter().zip(&self.weight_functions) {
            // Multiplication of `partial_derivatives` with the weight functions in
            // Fourier space (convolution in real space); summation leads to
            // functional derivative: the rows in the array are selected from the
            // running variable `k` with the number of rows needed for this
            // particular contribution
            let mut k = 0;

            // If local densities are present, their contributions are added directly
            if wf.local_density {
                functional_deriv_local += &pd.slice_axis(Axis(0), Slice::from(..wf.segments));
                k += wf.segments;
            }

            // Convolution of functional derivatives {scalar, component}
            for wf_i in &wf.scalar_component_weighted_densities {
                let pd_k = self.forward_transform_comps(
                    pd.slice_axis(Axis(0), Slice::from(k..k + wf.segments)),
                    None,
                );
                functional_deriv_k.add_assign(&(&pd_k * wf_i));
                k += wf.segments;
            }

            // Convolution of functional derivatives {vector, component}
            for wf_i in &wf.vector_component_weighted_densities {
                for (i, wf_i) in wf_i.outer_iter().enumerate() {
                    let pd_k = self.forward_transform_comps(
                        pd.slice_axis(Axis(0), Slice::from(k..k + wf.segments)),
                        Some(i),
                    );
                    functional_deriv_k.add_assign(&(pd_k * &wf_i));
                    k += wf.segments;
                }
            }

            // Convolution of functional derivatives {scalar, FMT}
            for wf_i in &wf.scalar_fmt_weighted_densities {
                let pd_k = self.forward_transform(pd.index_axis(Axis(0), k), None);
                functional_deriv_k.add_assign(&(wf_i * &pd_k));
                k += 1;
            }

            // Convolution of functional derivatives {vector, FMT}
            for wf_i in &wf.vector_fmt_weighted_densities {
                for (i, wf_i) in wf_i.outer_iter().enumerate() {
                    let pd_k = self.forward_transform(pd.index_axis(Axis(0), k), Some(i));
                    functional_deriv_k.add_assign(&(&wf_i * &pd_k));
                    k += 1;
                }
            }
        }

        // Backward transform of the non-local part of the functional derivative
        self.back_transform_comps(functional_deriv_k, functional_deriv.view_mut(), None);

        // Return sum over non-local and local contributions
        functional_deriv + functional_deriv_local
    }
}

/// The curvilinear convolver accounts for the shift that has to be performed
/// for spherical and polar transforms.
struct CurvilinearConvolver<T, D> {
    convolver: Rc<dyn Convolver<T, D>>,
    convolver_boundary: Rc<dyn Convolver<T, D>>,
}

impl<T, D: Dimension + RemoveAxis + 'static> CurvilinearConvolver<T, D>
where
    T: DctNum + ScalarOperand + DualNum<f64>,
    D::Larger: Dimension<Smaller = D>,
    D::Smaller: Dimension<Larger = D>,
    <D::Larger as Dimension>::Larger: Dimension<Smaller = D::Larger>,
{
    fn new(
        r: &Axis,
        z: &[&Axis],
        weight_functions: &[WeightFunctionInfo<T>],
        lanczos: Option<i32>,
    ) -> Rc<dyn Convolver<T, D>> {
        Rc::new(Self {
            convolver: ConvolverFFT::new(Some(r), z, weight_functions, lanczos),
            convolver_boundary: ConvolverFFT::new(None, z, weight_functions, lanczos),
        })
    }
}

impl<T, D: Dimension + RemoveAxis> Convolver<T, D> for CurvilinearConvolver<T, D>
where
    T: DctNum + ScalarOperand + DualNum<f64>,
    D::Smaller: Dimension<Larger = D>,
    D::Larger: Dimension<Smaller = D>,
{
    fn convolve(
        &self,
        mut profile: Array<T, D>,
        weight_function: &WeightFunction<T>,
    ) -> Array<T, D> {
        // subtract boundary profile from full profile
        let profile_boundary = profile
            .index_axis(Axis(0), profile.shape()[0] - 1)
            .into_owned();
        for mut lane in profile.outer_iter_mut() {
            lane.sub_assign(&profile_boundary);
        }

        // convolve full profile
        let mut result = self.convolver.convolve(profile, weight_function);

        // convolve boundary profile
        let profile_boundary = profile_boundary.insert_axis(Axis(0));
        let result_boundary = self
            .convolver_boundary
            .convolve(profile_boundary, weight_function);

        // Add boundary result back to full result
        let result_boundary = result_boundary.index_axis(Axis(0), 0);
        for mut lane in result.outer_iter_mut() {
            lane.add_assign(&result_boundary);
        }
        result
    }

    /// Calculates weighted densities via convolution from density profiles.
    fn weighted_densities(&self, density: &Array<T, D::Larger>) -> Vec<Array<T, D::Larger>> {
        // subtract boundary profile from full profile
        let density_boundary = density.index_axis(Axis(1), density.shape()[1] - 1);
        let mut density = density.to_owned();
        for mut lane in density.axis_iter_mut(Axis(1)) {
            lane.sub_assign(&density_boundary);
        }

        // convolve full profile
        let mut wd = self.convolver.weighted_densities(&density);

        // convolve boundary profile
        let density_boundary = density_boundary.insert_axis(Axis(1));
        let wd_boundary = self
            .convolver_boundary
            .weighted_densities(&density_boundary.to_owned());

        // Add boundary result back to full result
        for (wd, wd_boundary) in wd.iter_mut().zip(wd_boundary.iter()) {
            let wd_view = wd_boundary.index_axis(Axis(1), 0);
            for mut lane in wd.axis_iter_mut(Axis(1)) {
                lane.add_assign(&wd_view);
            }
        }

        wd
    }

    /// Calculates the functional derivative via convolution from partial derivatives
    /// of the Helmholtz energy functional.
    fn functional_derivative(
        &self,
        partial_derivatives: &[Array<T, D::Larger>],
        second_partial_derivatives: &[Array<T, <D::Larger as Dimension>::Larger>],
        weighted_densities: &[Array<T, D::Larger>],
    ) -> Array<T, D::Larger> {
        // subtract boundary profile from full profile
        let mut partial_derivatives_full = Vec::new();
        let mut partial_derivatives_boundary = Vec::new();
        for pd in partial_derivatives {
            let pd_boundary = pd.index_axis(Axis(1), pd.shape()[1] - 1).to_owned();
            let mut pd_full = pd.to_owned();
            for mut lane in pd_full.axis_iter_mut(Axis(1)) {
                lane.sub_assign(&pd_boundary);
            }
            partial_derivatives_full.push(pd_full);
            partial_derivatives_boundary.push(pd_boundary);
        }

        // convolve full profile
        let mut functional_derivative = self.convolver.functional_derivative(
            &partial_derivatives_full,
            &second_partial_derivatives,
            &weighted_densities,
        );

        // convolve boundary profile
        let mut partial_derivatives_boundary = Vec::new();
        for pd in partial_derivatives {
            let mut pd_boundary = pd.view();
            pd_boundary.collapse_axis(Axis(1), pd.shape()[1] - 1);
            partial_derivatives_boundary.push(pd_boundary.to_owned());
        }
        let functional_derivative_boundary = self.convolver_boundary.functional_derivative(
            &partial_derivatives_boundary,
            &second_partial_derivatives,
            &weighted_densities,
        );

        // Add boundary result back to full result
        let functional_derivative_view = functional_derivative_boundary.index_axis(Axis(1), 0);
        for mut lane in functional_derivative.axis_iter_mut(Axis(1)) {
            lane.add_assign(&functional_derivative_view);
        }

        functional_derivative
    }
}
