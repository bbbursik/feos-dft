use crate::profile::DFTSpecifications;
// use crate::PyDFTSpecification;
// use crate::python::PyDFTSpecification;
use numpy::PyArray1;
use pyo3::prelude::*;
// use PyDFTSpecification;

#[pyclass(name = "DFTSpecification", unsendable)]
#[pyo3(text_signature = "ChemicalPotential, Moles(Array1), TotalMoles(?)")]
#[derive(Clone)]
pub struct PyDFTSpecification(pub DFTSpecifications);

#[pymethods]
#[allow(non_snake_case)]
impl PyDFTSpecification {
    /// DFT with specified chemical potential.
    ///
    /// Returns
    /// -------
    /// DFTSpecification
    #[staticmethod]
    #[allow(non_snake_case)]
    #[pyo3(text_signature = "")]
    pub fn ChemicalPotential() -> Self {
        Self(DFTSpecifications::ChemicalPotential)
    }

    /// DFT with specified number of particles.
    ///
    /// Parameters
    /// ----------
    /// moles : PyArray
    ///     Number of particles for each component.
    ///
    /// Returns
    /// -------
    /// DFTSpecification
    #[staticmethod]
    #[allow(non_snake_case)]
    #[pyo3(text_signature = "(moles)")]
    pub fn Moles(moles: &PyArray1<f64>) -> Self {
        Self(DFTSpecifications::Moles {
            moles: moles.to_owned_array(),
        })
    }
    /// DFT with specified total number of moles and chemical potential differences.
    ///
    /// Parameters:
    /// -----------
    /// total_moles : float
    ///     Total number of particles.
    /// chemical_potential : PyArray
    ///     Chemical potential difference.
    ///
    /// Returns
    /// -------
    /// DFTSpecification
    #[staticmethod]
    #[allow(non_snake_case)]
    #[pyo3(text_signature = "(total_moles, chemical_potential)")]
    pub fn TotalMoles(total_moles: f64, chemical_potential: &PyArray1<f64>) -> Self {
        Self(DFTSpecifications::TotalMoles {
            total_moles,
            chemical_potential: chemical_potential.to_owned_array(),
        })
    }
}

#[macro_export]
macro_rules! impl_pore {
    ($func:ty, $py_func:ty) => {
        /// Parameters required to specify a 1D pore.
        ///
        /// Parameters
        /// ----------
        /// functional : HelmholtzEnergyFunctional
        ///     The Helmholtz energy functional.
        /// geometry : Geometry
        ///     The pore geometry.
        /// pore_size : SINumber
        ///     The width of the slit pore.
        /// potential : ExternalPotential
        ///     The potential used to model wall-fluid interactions.
        /// n_grid : int, optional
        ///     The number of grid points.
        /// potential_cutoff : float, optional
        ///     Maximum value for the external potential.
        ///
        /// Returns
        /// -------
        /// Pore1D
        ///
        #[pyclass(name = "Pore1D", unsendable)]
        #[pyo3(text_signature = "(functional, geometry, pore_size, potential, n_grid=None, potential_cutoff=None)")]
        pub struct PyPore1D(Pore1D<SIUnit, $func>);

        #[pyclass(name = "PoreProfile1D", unsendable)]
        pub struct PyPoreProfile1D(PoreProfile1D<SIUnit, $func>);

        impl_1d_profile!(PyPoreProfile1D, [get_r, get_z]);

        #[pymethods]
        impl PyPore1D {
            #[new]
            fn new(
                functional: &$py_func,
                geometry: PyGeometry,
                pore_size: PySINumber,
                potential: PyExternalPotential,
                n_grid: Option<usize>,
                potential_cutoff: Option<f64>,
            ) -> Self {
                Self(Pore1D::new(
                    &functional.0,
                    geometry.0,
                    pore_size.into(),
                    potential.0,
                    n_grid,
                    potential_cutoff,
                ))
            }

            /// Initialize the pore for the given bulk state.
            ///
            /// Parameters
            /// ----------
            /// bulk : State
            ///     The bulk state in equilibrium with the pore.
            /// external_potential : numpy.ndarray[float], optional
            ///     The external potential in the pore. Used to
            ///     save computation time in the case of costly
            ///     evaluations of external potentials.
            ///
            /// Returns
            /// -------
            /// PoreProfile1D
            #[pyo3(text_signature = "($self, bulk, external_potential=None, specification=None)")]
            fn initialize(
                &self,
                bulk: &PyState,
                external_potential: Option<&PyArray2<f64>>,
                specification: Option<PyDFTSpecification>
            ) -> PyResult<PyPoreProfile1D> {
                Ok(PyPoreProfile1D(self.0.initialize(
                    &bulk.0,
                    external_potential.map(|e| e.to_owned_array()).as_ref(),
                    specification.map(|s| s.0)
                )?))
            }
        }

        #[pymethods]
        impl PyPoreProfile1D {
              /// Create a new pore profile with a given specification.
            ///
            /// Parameters
            /// ----------
            /// specification: DFTSpecification
            ///
            #[pyo3(text_signature = "(specification)")]
            fn update_specification(&self, specification: PyDFTSpecification) -> Self {
                Self(self.0.update_specification(specification.0),
                )
            }


            #[getter]
            fn get_grand_potential(&self) -> Option<PySINumber> {
                self.0.grand_potential.map(PySINumber::from)
            }

            #[getter]
            fn get_interfacial_tension(&self) -> Option<PySINumber> {
                self.0.interfacial_tension.map(PySINumber::from)
            }
        }

        /// Parameters required to specify a 3D pore.
        ///
        /// Parameters
        /// ----------
        /// functional : HelmholtzEnergyFunctional
        ///     The Helmholtz energy functional.
        /// system_size : [SINumber; 3]
        ///     The size of the unit cell.
        /// n_grid : [int; 3]
        ///     The number of grid points in each direction.
        /// coordinates : numpy.ndarray[float]
        ///     The positions of all interaction sites in the solid.
        /// sigma_ss : numpy.ndarray[float]
        ///     The size parameters of all interaction sites.
        /// epsilon_k_ss : numpy.ndarray[float]
        ///     The energy parameter of all interaction sites.
        /// potential_cutoff: float, optional
        ///     Maximum value for the external potential.
        /// cutoff_radius: SINumber, optional
        ///     The cutoff radius for the calculation of solid-fluid interactions.
        /// l_grid: [SINumber;3], optional
        ///     length of the DFT domain; usually this is set as system_size, but in some cases helpful to set independently (e.g. if cutoff radius > system_size)
        ///
        /// Returns
        /// -------
        /// Pore3D
        ///
        #[pyclass(name = "Pore3D", unsendable)]
        #[pyo3(text_signature = "(functional, system_size, n_grid, coordinates, sigma_ss, epsilon_k_ss, potential_cutoff=None, cutoff_radius=None, l_grid=None)")]
        pub struct PyPore3D(Pore3D<SIUnit, $func>);

        #[pyclass(name = "PoreProfile3D", unsendable)]
        pub struct PyPoreProfile3D(PoreProfile3D<SIUnit, $func>);

        impl_3d_profile!(PyPoreProfile3D, get_x, get_y, get_z);

        #[pymethods]
        impl PyPore3D {
            #[new]
            fn new(
                functional: &$py_func,
                system_size: [PySINumber; 3],
                n_grid: [usize; 3],
                coordinates: &PySIArray2,
                sigma_ss: &PyArray1<f64>,
                epsilon_k_ss: &PyArray1<f64>,
                potential_cutoff: Option<f64>,
                cutoff_radius: Option<PySINumber>,
                l_grid: Option<[PySINumber; 3]>,

            ) -> Self {
                Self(Pore3D::new(
                    &functional.0,
                    [system_size[0].into(), system_size[1].into(), system_size[2].into()],
                    n_grid,
                    coordinates.clone().into(),
                    sigma_ss.to_owned_array(),
                    epsilon_k_ss.to_owned_array(),
                    potential_cutoff,
                    cutoff_radius.map(|c| c.into()),
                    l_grid.map(|c| [c[0].into(), c[1].into(), c[2].into()])
                ))
            }

            /// Initialize the pore for the given bulk state.
            ///
            /// Parameters
            /// ----------
            /// bulk : State
            ///     The bulk state in equilibrium with the pore.
            /// external_potential : numpy.ndarray[float], optional
            ///     The external potential in the pore. Used to
            ///     save computation time in the case of costly
            ///     evaluations of external potentials.
            ///
            /// Returns
            /// -------
            /// PoreProfile3D
            #[pyo3(text_signature = "($self, bulk, external_potential=None)")]
            fn initialize(
                &self,
                bulk: &PyState,
                external_potential: Option<&PyArray4<f64>>,
                specification: Option<PyDFTSpecification>
            ) -> PyResult<PyPoreProfile3D> {
                Ok(PyPoreProfile3D(self.0.initialize(
                    &bulk.0,
                    external_potential.map(|e| e.to_owned_array()).as_ref(),
                    specification.map(|s| s.0)
                )?))
            }
        }

        #[pymethods]
        impl PyPoreProfile3D {
            #[getter]
            fn get_grand_potential(&self) -> Option<PySINumber> {
                self.0.grand_potential.map(PySINumber::from)
            }

            #[getter]
            fn get_interfacial_tension(&self) -> Option<PySINumber> {
                self.0.interfacial_tension.map(PySINumber::from)
            }
        }
    };
}
