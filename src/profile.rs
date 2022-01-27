use crate::convolver::{Convolver, ConvolverFFT};
use crate::functional::{HelmholtzEnergyFunctional, DFT};
use crate::geometry::Grid;
use crate::solver::DFTSolver;
use crate::weight_functions::WeightFunctionInfo;
use feos_core::{Contributions, EosError, EosResult, EosUnit, EquationOfState, State};
use log::{info, warn};
use ndarray::{
    s, Array, Array1, Array3, ArrayBase, ArrayView, ArrayViewMut, ArrayViewMut1, Axis as Axis_nd,
    Data, Dimension, Ix1, Ix2, Ix3, RemoveAxis, Slice,
};
use num_dual::Dual64;
use num_dual::{HyperDual64, HyperDualVec, HyperDualVec64};
use quantity::{Quantity, QuantityArray, QuantityArray1, QuantityScalar};
use std::ops::{AddAssign, MulAssign};
use std::rc::Rc;

pub(crate) const MAX_POTENTIAL: f64 = 50.0;
pub(crate) const CUTOFF_RADIUS: f64 = 14.0;

/// General specifications for the chemical potential in a DFT calculation.
///
/// In the most basic case, the chemical potential is specified in a DFT calculation,
/// for more general systems, this trait provides the possibility to declare additional
/// equations for the calculation of the chemical potential during the iteration.
pub trait DFTSpecification<U, D: Dimension, F> {
    fn calculate_chemical_potential(
        &self,
        profile: &DFTProfile<U, D, F>,
        chemical_potential: &Array1<f64>,
        z: &Array1<f64>,
        bulk: &State<U, DFT<F>>,
    ) -> EosResult<Array1<f64>>;
}

/// Common specifications for the grand potentials in a DFT calculation.
pub enum DFTSpecifications {
    /// DFT with specified chemical potential.
    ChemicalPotential,
    /// DFT with specified number of particles.
    ///
    /// The solution is still a grand canonical density profile, but the chemical
    /// potentials are iterated together with the density profile to obtain a result
    /// with the specified number of particles.
    Moles { moles: Array1<f64> },
    /// DFT with specified total number of moles and chemical potential differences.
    TotalMoles {
        total_moles: f64,
        chemical_potential: Array1<f64>,
    },
}

impl DFTSpecifications {
    /// Calculate the number of particles from the profile.
    ///
    /// Call this after initializing the density profile to keep the number of
    /// particles constant in systems, where the number itself is difficult to obtain.
    pub fn moles_from_profile<U: EosUnit, D: Dimension, F: HelmholtzEnergyFunctional>(
        profile: &DFTProfile<U, D, F>,
    ) -> EosResult<Rc<Self>>
    where
        <D as Dimension>::Larger: Dimension<Smaller = D>,
    {
        let rho = profile.density.to_reduced(U::reference_density())?;
        Ok(Rc::new(Self::Moles {
            moles: profile.integrate_reduced_comp(&rho),
        }))
    }

    /// Calculate the number of particles from the profile.
    ///
    /// Call this after initializing the density profile to keep the total number of
    /// particles constant in systems, e.g. to fix the equimolar dividing surface.
    pub fn total_moles_from_profile<U: EosUnit, D: Dimension, F: HelmholtzEnergyFunctional>(
        profile: &DFTProfile<U, D, F>,
    ) -> EosResult<Rc<Self>>
    where
        <D as Dimension>::Larger: Dimension<Smaller = D>,
    {
        let rho = profile
            .density
            .to_reduced(U::reference_density())?
            .sum_axis(Axis_nd(0));
        let temperature = profile
            .bulk
            .temperature
            .to_reduced(U::reference_temperature())?;
        let mu_comp = profile
            .chemical_potential
            .to_reduced(U::reference_molar_energy())?
            / temperature;
        let mu_segment = profile.dft.component_index.mapv(|c| mu_comp[c]);
        Ok(Rc::new(Self::TotalMoles {
            total_moles: profile.integrate_reduced(rho),
            chemical_potential: mu_segment,
        }))
    }
}

impl<U: EosUnit, D: Dimension, F: HelmholtzEnergyFunctional> DFTSpecification<U, D, F>
    for DFTSpecifications
{
    fn calculate_chemical_potential(
        &self,
        profile: &DFTProfile<U, D, F>,
        chemical_potential: &Array1<f64>,
        z: &Array1<f64>,
        _: &State<U, DFT<F>>,
    ) -> EosResult<Array1<f64>> {
        let m = &profile.dft.m;
        Ok(match self {
            Self::ChemicalPotential => chemical_potential.clone(),
            Self::Moles { moles } => (moles / z).mapv(f64::ln) * m,
            Self::TotalMoles {
                total_moles,
                chemical_potential,
            } => {
                let exp_mu = (chemical_potential / m).mapv(f64::exp);
                ((&exp_mu * *total_moles) / (z * &exp_mu).sum()).mapv(f64::ln) * m
            }
        })
    }
}

/// A one-, two-, or three-dimensional density profile.
pub struct DFTProfile<U, D: Dimension, F> {
    pub grid: Grid,
    pub convolver: Rc<dyn Convolver<f64, D>>,
    pub convolver_wd: Rc<dyn Convolver<f64, D>>,
    pub dft: Rc<DFT<F>>,
    pub temperature: QuantityScalar<U>,
    pub density: QuantityArray<U, D::Larger>,
    pub chemical_potential: QuantityArray1<U>,
    pub specification: Rc<dyn DFTSpecification<U, D, F>>,
    pub external_potential: Array<f64, D::Larger>,
    pub bulk: State<U, DFT<F>>,
}

impl<U: EosUnit, F> DFTProfile<U, Ix1, F> {
    pub fn r(&self) -> QuantityArray1<U> {
        self.grid.grids()[0] * U::reference_length()
    }

    pub fn z(&self) -> QuantityArray1<U> {
        self.grid.grids()[0] * U::reference_length()
    }
}

impl<U: EosUnit, F> DFTProfile<U, Ix2, F> {
    pub fn edges(&self) -> (QuantityArray1<U>, QuantityArray1<U>) {
        (
            &self.grid.axes()[0].edges * U::reference_length(),
            &self.grid.axes()[1].edges * U::reference_length(),
        )
    }

    pub fn r(&self) -> QuantityArray1<U> {
        self.grid.grids()[0] * U::reference_length()
    }

    pub fn z(&self) -> QuantityArray1<U> {
        self.grid.grids()[1] * U::reference_length()
    }
}

impl<U: EosUnit, F> DFTProfile<U, Ix3, F> {
    pub fn edges(&self) -> (QuantityArray1<U>, QuantityArray1<U>, QuantityArray1<U>) {
        (
            &self.grid.axes()[0].edges * U::reference_length(),
            &self.grid.axes()[1].edges * U::reference_length(),
            &self.grid.axes()[2].edges * U::reference_length(),
        )
    }

    pub fn x(&self) -> QuantityArray1<U> {
        self.grid.grids()[0] * U::reference_length()
    }

    pub fn y(&self) -> QuantityArray1<U> {
        self.grid.grids()[1] * U::reference_length()
    }

    pub fn z(&self) -> QuantityArray1<U> {
        self.grid.grids()[2] * U::reference_length()
    }
}

impl<U: EosUnit, D: Dimension, F: HelmholtzEnergyFunctional> DFTProfile<U, D, F>
where
    <D as Dimension>::Larger: Dimension<Smaller = D>,
{
    /// Create a new density profile.
    ///
    /// If no external potential is specified, it is set to 0. The density is
    /// initialized based on the bulk state and the external potential. The
    /// specification is set to `ChemicalPotential` and needs to be overriden
    /// after this call if something else is required.
    pub fn new(
        grid: Grid,
        convolver: Rc<dyn Convolver<f64, D>>,
        convolver_wd: Rc<dyn Convolver<f64, D>>,
        bulk: &State<U, DFT<F>>,
        external_potential: Option<Array<f64, D::Larger>>,
    ) -> EosResult<Self> {
        let dft = bulk.eos.clone();

        // initialize external potential
        let external_potential = external_potential.unwrap_or_else(|| {
            let mut n_grid = vec![dft.component_index.len()];
            grid.axes()
                .iter()
                .for_each(|&ax| n_grid.push(ax.grid.len()));
            Array::zeros(n_grid).into_dimensionality().unwrap()
        });

        // intitialize density
        let t = bulk.temperature.to_reduced(U::reference_temperature())?;
        let isaft = dft
            .isaft_integrals(t, &external_potential, &convolver)
            .mapv(f64::abs)
            * (-&external_potential).mapv(f64::exp);
        let mut density = Array::zeros(external_potential.raw_dim());
        let bulk_density = bulk.partial_density.to_reduced(U::reference_density())?;
        for (s, &c) in dft.component_index.iter().enumerate() {
            density
                .index_axis_mut(Axis_nd(0), s)
                .assign(&(isaft.index_axis(Axis_nd(0), s).map(|is| is.min(1.0)) * bulk_density[c]));
        }

        Ok(Self {
            grid,
            convolver,
            convolver_wd,
            dft: bulk.eos.clone(),
            temperature: bulk.temperature,
            density: density * U::reference_density(),
            chemical_potential: bulk.chemical_potential(Contributions::Total),
            specification: Rc::new(DFTSpecifications::ChemicalPotential),
            external_potential,
            bulk: bulk.clone(),
        })
    }

    fn integrate_reduced(&self, mut profile: Array<f64, D>) -> f64 {
        let integration_weights = self.grid.integration_weights();

        for (i, w) in integration_weights.into_iter().enumerate() {
            for mut l in profile.lanes_mut(Axis_nd(i)) {
                l.mul_assign(w);
            }
        }
        profile.sum()
    }

    fn integrate_reduced_comp(&self, profile: &Array<f64, D::Larger>) -> Array1<f64> {
        Array1::from_shape_fn(profile.shape()[0], |i| {
            self.integrate_reduced(profile.index_axis(Axis_nd(0), i).to_owned())
        })
    }

    /// Return the volume of the profile.
    ///
    /// Depending on the geometry, the result is in m, m² or m³.
    pub fn volume(&self) -> QuantityScalar<U> {
        self.grid
            .axes()
            .iter()
            .fold(None, |acc, &ax| {
                Some(acc.map_or(ax.volume(), |acc| acc * ax.volume()))
            })
            .unwrap()
    }

    /// Integrate a given profile over the iteration domain.
    pub fn integrate<S: Data<Elem = f64>>(
        &self,
        profile: &Quantity<ArrayBase<S, D>, U>,
    ) -> QuantityScalar<U> {
        profile.integrate(&self.grid.integration_weights_unit())
    }

    /// Integrate each component individually.
    pub fn integrate_comp<S: Data<Elem = f64>>(
        &self,
        profile: &Quantity<ArrayBase<S, D::Larger>, U>,
    ) -> QuantityArray1<U> {
        QuantityArray1::from_shape_fn(profile.shape()[0], |i| {
            self.integrate(&profile.index_axis(Axis_nd(0), i))
        })
    }

    /// Return the number of moles of each component in the system.
    pub fn moles(&self) -> QuantityArray1<U> {
        let rho = self.density.to_reduced(U::reference_density()).unwrap();
        let mut d = rho.raw_dim();
        d[0] = self.dft.components();
        let mut density_comps = Array::zeros(d);
        for (i, &j) in self.dft.component_index.iter().enumerate() {
            density_comps
                .index_axis_mut(Axis_nd(0), j)
                .assign(&rho.index_axis(Axis_nd(0), i));
        }
        self.integrate_comp(&(density_comps * U::reference_density()))
    }

    /// Return the total number of moles in the system.
    pub fn total_moles(&self) -> QuantityScalar<U> {
        self.moles().sum()
    }
}

impl<U: Clone, D: Dimension, F> Clone for DFTProfile<U, D, F> {
    fn clone(&self) -> Self {
        Self {
            grid: self.grid.clone(),
            convolver: self.convolver.clone(),
            convolver_wd: self.convolver_wd.clone(),
            dft: self.dft.clone(),
            temperature: self.temperature.clone(),
            density: self.density.clone(),
            chemical_potential: self.chemical_potential.clone(),
            specification: self.specification.clone(),
            external_potential: self.external_potential.clone(),
            bulk: self.bulk.clone(),
        }
    }
}

impl<U, F> DFTProfile<U, Ix1, F>
where
    U: EosUnit,
    F: HelmholtzEnergyFunctional,
{
    pub fn local_functional_derivative_v2(&self) -> EosResult<Vec<Array<f64, Ix2>>> {
        let temperature = self.temperature.to_reduced(U::reference_temperature())?;
        // println!("Version of 12:01");
        let densities = self.density.to_reduced(U::reference_density())?; //.view()
        let dx = self.grid.grids()[0][1] - self.grid.grids()[0][0];

        let k0 = HyperDual64::from(0.0).derive1().derive2();

        let weighted_densities = self.local_weighted_densities()?;
        let contributions = self.dft.functional.contributions();
        let mut functional_derivative_0 = Array::zeros(densities.raw_dim())
            .into_dimensionality()
            .unwrap();
        let mut functional_derivative_1 = Array::zeros(densities.raw_dim())
            .into_dimensionality()
            .unwrap();
        let mut functional_derivative_2 = Array::zeros(densities.raw_dim())
            .into_dimensionality()
            .unwrap();
        for (c, wd) in contributions.iter().zip(weighted_densities) {
            let wf = c.weight_functions(HyperDual64::from(temperature));
            let w = wf.weight_constants(k0, 1);
            let w0 = w.mapv(|w| w.re);
            let w1 = w.mapv(|w| -w.eps1[0]);
            let w2 = w.mapv(|w| -0.5 * w.eps1eps2[(0, 0)]);

            // println!("w0 = {:?}", w0);
            // println!("w1 = {:?}", w1);
            // println!("w2 = {:?}", w2);

            let segments = wf.component_index.len();
            let nwd = wd.shape()[0];
            let ngrid = wd.len() / nwd;
            let mut dim = vec![nwd, nwd];
            wd.shape().iter().skip(1).for_each(|&d| dim.push(d));

            let mut phi = Array::zeros(densities.raw_dim().remove_axis(Axis_nd(0)));
            let mut first_partial_derivative = Array::zeros(wd.raw_dim());
            let mut second_partial_derivative: Array<_, Ix3> =
                Array::zeros(dim).into_dimensionality().unwrap();
            //let mut spd = Array::zeros(wd.raw_dim());
            let grad_weighted_density = self.gradient(wd.view(), dx)?;
            let lapl_weighted_density = self.gradient(grad_weighted_density.view(), dx)?;

            c.second_partial_derivatives(
                temperature,
                wd.into_shape((nwd, ngrid)).unwrap(),
                phi.view_mut().into_shape(ngrid).unwrap(),
                first_partial_derivative
                    .view_mut()
                    .into_shape((nwd, ngrid))
                    .unwrap(),
                second_partial_derivative
                    .view_mut()
                    .into_shape((nwd, nwd, ngrid))
                    .unwrap(),
            )?;

            // calculate gradients of partial derivatives
            // !! MAKES SENSE ONLY IN 1D FOR NOW!! even though it should compile
            let grad_first_partial_derivative =
                self.gradient(first_partial_derivative.view(), dx)?;
            let lapl_first_partial_derivative =
                self.gradient(grad_first_partial_derivative.view(), dx)?;
            let mut grad_second_partial_derivative =
                Array::zeros(second_partial_derivative.raw_dim());
            // let mut grad_second_partial_derivative: Array3<f64> =
            //     Array::zeros(second_partial_derivative.raw_dim());

            for (spd, mut res) in second_partial_derivative
                .outer_iter()
                .zip(grad_second_partial_derivative.outer_iter_mut())
            {
                res.assign(&self.gradient(spd, dx)?);
            }

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
                functional_derivative_0 +=
                    &first_partial_derivative.slice_axis(Axis_nd(0), Slice::from(..segments));
                k += segments;
            }

            // Calculating functional derivative {scalar, component}
            for wf_i in &wf.scalar_component_weighted_densities {
                for (
                    i,
                    ((fpd, spds), gradients_spd), // mut res0), //, mut res2), //((((((fpd, spds), gradients_spd), grad_wd), lapl_wd), mut res0), mut res2),
                ) in first_partial_derivative
                    .slice_axis(Axis_nd(0), Slice::from(k..k + segments))
                    .outer_iter()
                    .zip(
                        second_partial_derivative
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
                for (i, (spds, mut res1)) in second_partial_derivative
                    .slice_axis(Axis_nd(0), Slice::from(k..k + segments))
                    .outer_iter()
                    .zip(functional_derivative_1.outer_iter_mut())
                    .enumerate()
                {
                    for (spd, grad_wd) in spds.outer_iter().zip(grad_weighted_density.outer_iter())
                    {
                        res1.add_assign(
                            &(-&spd
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
                    res0.add_assign(
                        &(&first_partial_derivative.index_axis(Axis_nd(0), k)
                            * w0.slice(s![k, ..])[i]),
                    );

                    for (((spd, grad_spd), grad_wd), lapl_wd) in second_partial_derivative
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
                    for (spd, grad_wd) in second_partial_derivative
                        .index_axis(Axis_nd(0), k)
                        .outer_iter()
                        .zip(grad_weighted_density.outer_iter())
                    {
                        res1.add_assign(&(-&spd * &grad_wd * w1.slice(s![k, ..])[i]));
                    }
                }
                k += 1;
            }
        }

        Ok(vec![
            functional_derivative_0,
            functional_derivative_1,
            functional_derivative_2,
        ])
    }

    pub fn local_functional_derivative(&self) -> EosResult<Vec<Array<f64, Ix2>>> {
        let temperature = self.temperature.to_reduced(U::reference_temperature())?;
        // println!("Version of 8:50");
        let densities = self.density.to_reduced(U::reference_density())?; //.view()
        let dx = self.grid.grids()[0][1] - self.grid.grids()[0][0];

        let k0 = HyperDual64::from(0.0).derive1().derive2();

        let weighted_densities = self.weighted_densities()?;
        let contributions = self.dft.functional.contributions();
        let mut functional_derivative_0 = Array::zeros(densities.raw_dim())
            .into_dimensionality()
            .unwrap();
        let mut functional_derivative_1 = Array::zeros(densities.raw_dim())
            .into_dimensionality()
            .unwrap();
        let mut functional_derivative_2 = Array::zeros(densities.raw_dim())
            .into_dimensionality()
            .unwrap();
        for (c, wd) in contributions.iter().zip(weighted_densities) {
            let wf = c.weight_functions(HyperDual64::from(temperature));
            let w = wf.weight_constants(k0, 1);
            let w0 = w.mapv(|w| w.re);
            let w1 = w.mapv(|w| -w.eps1[0]);
            let w2 = w.mapv(|w| -0.5 * w.eps1eps2[(0, 0)]);

            // println!("w0 = {:?}", w0);
            // println!("w1 = {:?}", w1);
            // println!("w2 = {:?}", w2);

            let segments = wf.component_index.len();
            let nwd = wd.shape()[0];
            let ngrid = wd.len() / nwd;
            let mut phi = Array::zeros(densities.raw_dim().remove_axis(Axis_nd(0)));
            let mut first_partial_derivative = Array::zeros(wd.raw_dim());
            //let mut spd = Array::zeros(wd.raw_dim());
            c.first_partial_derivatives(
                temperature,
                wd.into_shape((nwd, ngrid)).unwrap(),
                phi.view_mut().into_shape(ngrid).unwrap(),
                first_partial_derivative
                    .view_mut()
                    .into_shape((nwd, ngrid))
                    .unwrap(),
            )?;

            // calculate gradients of partial derivatives
            // !! MAKES SENSE ONLY IN 1D FOR NOW!! even though it should compile
            let grad_first_partial_derivative =
                self.gradient(first_partial_derivative.view(), dx)?;
            let lapl_first_partial_derivative =
                self.gradient(grad_first_partial_derivative.view(), dx)?;

            // Initilaizing row index for non-local functional derivative
            let mut k = 0;

            // Assigning possible local densities to the front of the array
            if wf.local_density {
                functional_derivative_0 +=
                    &first_partial_derivative.slice_axis(Axis_nd(0), Slice::from(..segments));
                k += segments;
            }

            // Calculating functional derivative {scalar, component}
            for wf_i in &wf.scalar_component_weighted_densities {
                for (i, (((fpd, lapl), mut res0), mut res2)) in first_partial_derivative
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
                    res1.add_assign(&(-&grad * (w1.slice(s![k..k + segments, ..]).into_diag()[i])));
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
                    res0.add_assign(
                        &(&first_partial_derivative.index_axis(Axis_nd(0), k)
                            * w0.slice(s![k, ..])[i]),
                    );
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
                        &(-&grad_first_partial_derivative.index_axis(Axis_nd(0), k)
                            * w1.slice(s![k, ..])[i]),
                    );
                }
                k += 1;
            }
        }

        Ok(vec![
            functional_derivative_0,
            functional_derivative_1,
            functional_derivative_2,
        ])
    }

    pub fn gradient(&self, f: ArrayView<f64, Ix2>, dx: f64) -> EosResult<Array<f64, Ix2>> {
        let grad = Array::from_shape_fn(f.raw_dim(), |(c, i)| {
            let d = if i == 0 {
                2.0 * (f[(c, 1)] - f[(c, 0)]) // Left value --> where from?
            } else if i == f.shape()[1] - 1 {
                2.0 * (f[(c, f.shape()[1] - 1)] - f[(c, f.shape()[1] - 2)])
            } else {
                f[(c, i + 1)] - f[(c, i - 1)]
            };
            d / (2.0 * dx)
        });
        Ok(grad)
    }

    pub fn local_weighted_densities(&self) -> EosResult<Vec<Array<f64, Ix2>>> {
        let densities = self.density.to_reduced(U::reference_density())?; //.view()
        let dx = self.grid.grids()[0][1] - self.grid.grids()[0][0];

        let gradient = self.gradient(densities.view(), dx)?;
        let laplace = self.gradient(gradient.view(), dx)?;
        let temperature =
            HyperDual64::from(self.temperature.to_reduced(U::reference_temperature())?);

        let weight_functions: Vec<WeightFunctionInfo<HyperDual64>> = self
            .dft
            .functional
            .contributions()
            .iter()
            .map(|c| c.weight_functions(temperature))
            .collect();

        let k0 = HyperDual64::from(0.0).derive1().derive2();
        let mut weighted_densities_vec = Vec::with_capacity(weight_functions.len());

        //loop over contributions
        for wf in weight_functions.iter() {
            let segments = wf.component_index.len();

            let w = wf.weight_constants(k0, 1); //can rewrite this with the corresponding function for the weight constants (i.e. scalar_weigth_const)
            let w0 = w.mapv(|w| w.re);
            let w1 = w.mapv(|w| -w.eps1[0]);
            let w2 = w.mapv(|w| -0.5 * w.eps1eps2[(0, 0)]);

            // number of weighted densities
            let n_wd = wf.n_weighted_densities(1);

            // Allocating new array for intended weighted densities
            let mut dim = vec![n_wd];
            densities.shape().iter().skip(1).for_each(|&d| dim.push(d));

            let mut weighted_densities = Array::zeros(dim).into_dimensionality().unwrap();

            // Initilaizing row index for non-local weighted densities
            let mut k = 0;

            // Assigning possible local densities to the front of the array
            if wf.local_density {
                weighted_densities
                    .slice_axis_mut(Axis_nd(0), Slice::from(0..segments))
                    .assign(&densities);
                k += segments;
            }

            // Calculating weighted densities {scalar, component}
            for wf_i in &wf.scalar_component_weighted_densities {
                for (i, ((rho, lapl), mut res)) in densities
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
                for (i, (rho, lapl)) in densities.outer_iter().zip(laplace.outer_iter()).enumerate()
                {
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
            weighted_densities_vec.push(weighted_densities);
        }

        Ok(weighted_densities_vec)
    }
}

impl<U, D, F> DFTProfile<U, D, F>
where
    U: EosUnit,
    D: Dimension,
    D::Larger: Dimension<Smaller = D>,
    F: HelmholtzEnergyFunctional,
{
    
    // pub fn weight_constants(&self) -> Vec<Array<HyperDual64, D::Larger>> {
    //     self
    //         .convolver
    //         .weight_constants
    // }
    pub fn weighted_densities(&self) -> EosResult<Vec<Array<f64, D::Larger>>> {
        Ok(self
            .convolver
            .weighted_densities(&self.density.to_reduced(U::reference_density())?))
    }

    pub fn weighted_densities_FFT(&self) -> EosResult<Vec<Array<f64, D::Larger>>> {
        Ok(self
            .convolver_wd
            .weighted_densities(&self.density.to_reduced(U::reference_density())?))
    }

    pub fn functional_derivative(&self) -> EosResult<Array<f64, D::Larger>> {
        let (_, dfdrho) = self.dft.functional_derivative(
            self.temperature.to_reduced(U::reference_temperature())?,
            &self.density.to_reduced(U::reference_density())?,
            &self.convolver,
            &self.convolver_wd,
        )?;
        Ok(dfdrho)
    }

    #[allow(clippy::type_complexity)]
    pub fn residual(&self, log: bool) -> EosResult<(Array<f64, D::Larger>, Array1<f64>)> {
        // Read from profile
        let temperature = self.temperature.to_reduced(U::reference_temperature())?;
        let density = self.density.to_reduced(U::reference_density())?;
        let mu_comp = self
            .chemical_potential
            .to_reduced(U::reference_molar_energy())?
            / temperature;
        let chemical_potential = self.dft.component_index.mapv(|i| mu_comp[i]);
        let mut bulk = self.bulk.clone();

        // Allocate residual vectors
        let mut res_rho = Array::zeros(density.raw_dim());
        let mut res_mu = Array1::zeros(chemical_potential.len());

        self.calculate_residual(
            temperature,
            &density,
            &chemical_potential,
            &mut bulk,
            res_rho.view_mut(),
            res_mu.view_mut(),
            log,
        )?;

        Ok((res_rho, res_mu))
    }

    fn calculate_residual(
        &self,
        temperature: f64,
        density: &Array<f64, D::Larger>,
        chemical_potential: &Array1<f64>,
        bulk: &mut State<U, DFT<F>>,
        mut res_rho: ArrayViewMut<f64, D::Larger>,
        mut res_mu: ArrayViewMut1<f64>,
        log: bool,
    ) -> EosResult<()> {
        // Update bulk state
        let mut mu_comp = Array::zeros(bulk.eos.components());
        for (s, &c) in self.dft.component_index.iter().enumerate() {
            mu_comp[c] = chemical_potential[s];
        }
        bulk.update_chemical_potential(&(mu_comp * temperature * U::reference_molar_energy()))?;

        // calculate intrinsic functional derivative
        let (_, mut dfdrho) = self.dft.functional_derivative(
            temperature,
            density,
            &self.convolver,
            &self.convolver_wd,
        )?;

        // calculate total functional derivative
        dfdrho += &self.external_potential;

        // calculate isaft integrals
        let isaft = self
            .dft
            .isaft_integrals(temperature, &dfdrho, &self.convolver);

        // Euler-Lagrange equation
        let m = &self.dft.m;
        res_rho
            .outer_iter_mut()
            .zip(dfdrho.outer_iter())
            .zip(chemical_potential.iter())
            .zip(m.iter())
            .zip(density.outer_iter())
            .zip(isaft.outer_iter())
            .for_each(|(((((mut res, df), &mu), &m), rho), is)| {
                res.assign(
                    &(if log {
                        rho.mapv(f64::ln) - (mu - &df) / m - is.mapv(f64::ln)
                    } else {
                        &rho - &(((mu - &df) / m).mapv(f64::exp) * is)
                    }),
                );
            });

        // set residual to 0 where external potentials are overwhelming
        res_rho
            .iter_mut()
            .zip(self.external_potential.iter())
            .for_each(|(r, &p)| {
                if p + f64::EPSILON >= MAX_POTENTIAL {
                    *r = 0.0;
                }
            });

        // Additional residuals for the calculation of the chemical potential
        let z: Array1<_> = dfdrho
            .outer_iter()
            .zip(m.iter())
            .zip(isaft.outer_iter())
            .map(|((df, &m), is)| self.integrate_reduced((-&df / m).mapv(f64::exp) * is))
            .collect();
        let mu_spec =
            self.specification
                .calculate_chemical_potential(self, chemical_potential, &z, bulk)?;

        res_mu.assign(
            &(if log {
                chemical_potential - &mu_spec
            } else {
                chemical_potential.mapv(f64::exp) - mu_spec.mapv(f64::exp)
            }),
        );

        Ok(())
    }

    pub fn solve(&mut self, solver: Option<&DFTSolver>, debug: bool) -> EosResult<()> {
        // unwrap solver
        let solver = solver.cloned().unwrap_or_default();

        // Read from profile
        let temperature = self.temperature.to_reduced(U::reference_temperature())?;
        let mut density = self.density.to_reduced(U::reference_density())?;
        let mut mu_comp = self
            .chemical_potential
            .to_reduced(U::reference_molar_energy())?
            / temperature;
        let mut chemical_potential = self.dft.component_index.mapv(|i| mu_comp[i]);
        let mut bulk = self.bulk.clone();

        // initialize x-vector
        let n_rho = density.len();
        let mut x = Array1::zeros(n_rho + density.shape()[0]);
        x.slice_mut(s![..n_rho])
            .assign(&density.view().into_shape(n_rho).unwrap());
        x.slice_mut(s![n_rho..])
            .assign(&chemical_potential.mapv(f64::exp));

        // Residual function
        let mut residual =
            |x: &Array1<f64>, mut res: ArrayViewMut1<f64>, log: bool| -> EosResult<()> {
                // Read density and chemical potential from solution vector
                density.assign(&x.slice(s![..n_rho]).into_shape(density.shape()).unwrap());
                chemical_potential.assign(&x.slice(s![n_rho..]).mapv(f64::ln));

                // Create views for different residuals
                let (res_rho, res_mu) = res.multi_slice_mut((s![..n_rho], s![n_rho..]));
                let res_rho = res_rho.into_shape(density.raw_dim()).unwrap();

                // Calculate residual
                self.calculate_residual(
                    temperature,
                    &density,
                    &chemical_potential,
                    &mut bulk,
                    res_rho,
                    res_mu,
                    log,
                )
            };

        // Call solver(s)
        let (converged, iterations) = solver.solve(&mut x, &mut residual)?;
        if converged {
            info!("DFT solved in {} iterations", iterations);
        } else if debug {
            warn!("DFT not converged in {} iterations", iterations);
        } else {
            return Err(EosError::NotConverged(String::from("DFT")));
        }

        // Update profile
        self.density = density * U::reference_density();
        for (s, &c) in self.dft.component_index.iter().enumerate() {
            mu_comp[c] = chemical_potential[s];
        }
        self.chemical_potential = mu_comp * temperature * U::reference_molar_energy();
        self.bulk = bulk;

        Ok(())
    }
}

impl<U: EosUnit, D: Dimension + RemoveAxis + 'static, F: HelmholtzEnergyFunctional>
    DFTProfile<U, D, F>
where
    D::Larger: Dimension<Smaller = D>,
    D::Smaller: Dimension<Larger = D>,
    <D::Larger as Dimension>::Larger: Dimension<Smaller = D::Larger>,
{
    pub fn entropy_density(&self, contributions: Contributions) -> EosResult<QuantityArray<U, D>> {
        // initialize convolver
        let t = self.temperature.to_reduced(U::reference_temperature())?;
        let functional_contributions = self.dft.functional.contributions();
        let weight_functions: Vec<WeightFunctionInfo<Dual64>> = functional_contributions
            .iter()
            .map(|c| c.weight_functions(Dual64::from(t).derive()))
            .collect();
        let convolver = ConvolverFFT::plan(&self.grid, &weight_functions, None);

        Ok(self.dft.entropy_density(
            t,
            &self.density.to_reduced(U::reference_density())?,
            &convolver,
            contributions,
        )? * (U::reference_entropy() / U::reference_volume()))
    }

    pub fn entropy(&self, contributions: Contributions) -> EosResult<QuantityScalar<U>> {
        Ok(self.integrate(&self.entropy_density(contributions)?))
    }

    pub fn internal_energy(&self, contributions: Contributions) -> EosResult<QuantityScalar<U>> {
        // initialize convolver
        let t = self.temperature.to_reduced(U::reference_temperature())?;
        let functional_contributions = self.dft.functional.contributions();
        let weight_functions: Vec<WeightFunctionInfo<Dual64>> = functional_contributions
            .iter()
            .map(|c| c.weight_functions(Dual64::from(t).derive()))
            .collect();
        let convolver = ConvolverFFT::plan(&self.grid, &weight_functions, None);

        let internal_energy_density = self.dft.internal_energy_density(
            t,
            &self.density.to_reduced(U::reference_density())?,
            &self.external_potential,
            &convolver,
            contributions,
        )? * U::reference_pressure();
        Ok(self.integrate(&internal_energy_density))
    }
}
