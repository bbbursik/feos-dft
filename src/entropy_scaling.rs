use feos_core::{EosResult, EosUnit};
use feos_dft::fundamental_measure_theory::{FMTContribution, FMTProperties};
use feos_dft::{
    FunctionalContribution, HelmholtzEnergyFunctional, WeightFunction, WeightFunctionInfo,
    WeightFunctionShape,
};
use ndarray::{Array, Dimension};
use quantity::{QuantityArray, QuantityScalar};

pub trait EntropyScalingFunctionalContribution: FunctionalContribution {
    fn weight_functions_entropy(&self, temperature: f64) -> WeightFunctionInfo<f64>;
}

impl<P: FMTProperties> EntropyScalingFunctionalContribution for FMTContribution<P> {
    fn weight_functions_entropy(&self, temperature: f64) -> WeightFunctionInfo<f64> {
        let r = self.properties.hs_diameter(temperature) * 0.5;
        WeightFunctionInfo::new(self.properties.n_segments().clone(), false).add(
            WeightFunction::new_scaled(r, WeightFunctionShape::Theta),
            true,
        )
    }
}

pub trait EntropyScalingFunctional<U: EosUnit>: HelmholtzEnergyFunctional {
    fn entropy_scaling_contributions(&self) -> &[Box<dyn EntropyScalingFunctionalContribution>];

    /// Viscosity referaence for entropy scaling for the shear viscosity.
    fn viscosity_reference<D>(
        &self,
        density: &QuantityArray<U, D::Larger>,
        temperature: QuantityScalar<U>,
    ) -> EosResult<QuantityArray<U, D>>
    where
        D: Dimension,
        D::Larger: Dimension<Smaller = D>;

    /// Correlation function for entropy scaling of the shear viscosity.
    fn viscosity_correlation<D>(
        &self,
        s_res: &Array<f64, D>,
        density: &QuantityArray<U, D::Larger>,
    ) -> EosResult<Array<f64, D>>
    where
        D: Dimension,
        D::Larger: Dimension<Smaller = D>;

    // /// Self-diffusion references for entropy scaling.
    // fn diffusion_reference<D>(
    //     &self,
    //     density: &QuantityArray<U, D::Larger>,
    //     temperature: QuantityScalar<U>,
    // ) -> EosResult<QuantityArray<U, D::Larger>>
    // where
    //     D: Dimension,
    //     D::Larger: Dimension<Smaller = D>;

    // /// Correlation functions for entropy scaling of self-diffusion coefficients.
    // fn diffusion_correlation<D>(
    //     &self,
    //     s_res: &Array<f64, D>,
    //     density: &QuantityArray<U, D::Larger>,
    // ) -> EosResult<Array<f64, D::Larger>>
    // where
    //     D: Dimension,
    //     D::Larger: Dimension<Smaller = D>;
}

impl<U, D, F> DFTProfile<U, D, F>
where
    U: EosUnit,
    D: Dimension,
    D::Larger: Dimension<Smaller = D>,
    F: HelmholtzEnergyFunctional,
{
    pub fn viscosity_profile(&self) -> EosResult<Vec<Array<f64, D>>> {
        let s_res = self
            .dft
            .functional
            .entropy_density_contributions()?
            .to_reduced(U::reference_moles().powi(-1))?
            .mapv(|v| f64::min(v, 0.0));

        // let visc_ref = self
        //     .dft
        //     .functional
        //     .viscosity_reference::<Ix1>(density, self.temperature)
        //     .unwrap();

        let mut viscosity_shear = Array1::zeros(points_total);

        // viscosity_shear.slice_mut(s![2..-2]).assign(
        //     &(self
        //         .dft
        //         .functional
        //         .viscosity_correlation::<Ix1>(&s_res, density)
        //         .unwrap()
        //         .mapv(f64::exp)
        //         * visc_ref.to_reduced(U::reference_viscosity())?),
        // );

        viscosity_shear = viscosity_shear * U::reference_viscosity();

        Ok(viscosity_shear)
    }
}
