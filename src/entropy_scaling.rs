use crate::fundamental_measure_theory::{FMTContribution, FMTProperties, FMTFunctional};
use crate::profile::DFTProfile;
use crate::{
    ConvolverFFT, FunctionalContribution, HelmholtzEnergyFunctional, WeightFunction,
    WeightFunctionInfo, WeightFunctionShape,
};
use feos_core::{Contributions, EosResult, EosUnit};
use ndarray::{Array, Dimension, Axis, RemoveAxis};
use num_dual::Dual64;
use quantity::{QuantityArray, QuantityScalar};

pub trait EntropyScalingFunctionalContribution: FunctionalContribution {
    fn weight_functions_entropy(&self, temperature: f64) -> WeightFunctionInfo<f64>;
}

impl<P: FMTProperties> EntropyScalingFunctionalContribution for FMTContribution<P> {
    fn weight_functions_entropy(&self, temperature: f64) -> WeightFunctionInfo<f64> {
        let r = self.properties.hs_diameter(temperature) * 0.5;
        WeightFunctionInfo::new(self.properties.component_index(), false).add(
            WeightFunction::new_scaled(r, WeightFunctionShape::Theta),
            true,
        )
    }
}


impl EntropyScalingFunctionalContribution for FMTFunctional {
    fn weight_functions_entropy(&self, temperature: f64) -> WeightFunctionInfo<f64> {
        let r = self.properties.hs_diameter(temperature) * 0.5;
        WeightFunctionInfo::new(self.properties.component_index(), false).add(
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
    F: HelmholtzEnergyFunctional + EntropyScalingFunctional<U>,
{
    pub fn viscosity_profile(&self) -> EosResult<QuantityArray<U, D>> {
        let temperature_red = self.temperature.to_reduced(U::reference_temperature())?;
        let density_red = self.density.to_reduced(U::reference_density())?;

        let weight_functions_dual: Vec<WeightFunctionInfo<Dual64>> = self
            .dft
            .functional
            .contributions()
            .iter()
            .map(|c| c.weight_functions(Dual64::from(temperature_red).derive()))
            .collect();

        let convolver_dual = ConvolverFFT::plan(&self.grid, &weight_functions_dual, None);

        // Initialize entropy convolver
        let functional_contributions_entropy = self.dft.functional.entropy_scaling_contributions();
        let weight_functions_entropy: Vec<WeightFunctionInfo<f64>> =
            functional_contributions_entropy
                .iter()
                .map(|c| c.weight_functions_entropy(temperature_red))
                .collect();
        let convolver_entropy = ConvolverFFT::plan(
            &self.grid,
            &weight_functions_entropy,
            None,
        );


        // Code (originally placed in entropy_density_contributions)

        // Weighted densities
        let weighted_densities_entropy = convolver_entropy.weighted_densities(&density_red);


            // Molar entropy calculation for each contribution
        let entropy_molar_contributions = self
            .dft
            .entropy_density_contributions(
                temperature_red,
                &density_red,
                &convolver_dual,
                Contributions::Residual,
            )?
            .iter()
            .zip(weighted_densities_entropy.iter())
            .map(|(v, w)| {
                (v * U::reference_volume().powi(-1)
                    / (&w.index_axis(Axis(0), 0) * U::reference_density()))
                    // / (&w.slice(s![0, ..]) * U::reference_density()))
                .to_reduced(U::reference_moles().powi(-1))
                .unwrap()
            })
            .collect::<Vec<_>>();


        let mut dim = vec![];
        self.density.shape().iter().skip(0).for_each(|&d| dim.push(d));
        let mut entropy_molar = Array::zeros(dim);
        // let mut entropy_molar = Array::zeros(self.density.raw_dim().remove_axis(Axis(0)));
        for contr in entropy_molar_contributions.iter() {
            entropy_molar += contr;
        }

        // 
        // entropy_molar = entropy_molar * U::reference_moles().powi(-1);
        let s_res = entropy_molar.mapv(|s| f64::min(s, 0.0));

      

        // let s_res = self
        //     .dft
        //     // .functional
        //     .entropy_density_contributions(
        //         temperature,
        //         &density,
        //         &self.convolver,
        //         Contributions::Residual,
        //     )?
        //     .to_reduced(U::reference_moles().powi(-1))?
        //     .mapv(|v| f64::min(v, 0.0));

        let visc_ref = self
            .dft
            .functional
            .viscosity_reference(&self.density, self.temperature)
            .unwrap();

        // let mut viscosity_shear = Array::zeros(entropy_molar.raw_dim());
        let mut viscosity_shear = self.dft.functional.viscosity_correlation(&s_res, &self.density).unwrap().mapv(f64::exp) * visc_ref.to_reduced(U::reference_viscosity())?;


        // viscosity_shear.slice_mut(s![2..-2]).assign( 
        //     &(self
        //         .dft
        //         .functional
        //         .viscosity_correlation::<Ix1>(&s_res, density)
        //         .unwrap()
        //         .mapv(f64::exp)
        //         * visc_ref.to_reduced(U::reference_viscosity())?),
        // );


        Ok(viscosity_shear * U::reference_viscosity())
    
    }
}
