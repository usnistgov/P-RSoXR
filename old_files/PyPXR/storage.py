class PXR_MGMixedSlab(PXR_Component):
    """
    A slab component made of several components
    **Not confirmed by developer. Still in active testing/development.

    Parameters
    ----------
    thick : refnx.analysis.Parameter or float
        thickness of slab (Angstrom)
    sld_list : sequence of {refnx.reflect.Scatterer, complex, float}
        Sequence of (complex) SLDs that are contained in film
        (/1e-6 Angstrom**2) **The first in the list is the 'host' matrix for Maxwell-Garnett approximation
    vf : float
        Volume fraction of the dopant material in the 2-component slab
    shapefunc : List
        Three floats representing the x, y, z shape parameters for anisotropic model
        **No guarentee by the developer that one will be more 'correct' than another
        Spherical inclusion: [1/3, 1/3, 1/3] ***Default
        Rod-like inclusion: [1/2, 1/2, 0]
        Disk-like inclusion: []
        
    rough : refnx.analysis.Parameter or float
        roughness on top of this slab (Angstrom)
    name : str
        Name of this slab

    Notes
    -----
    The SLD of this Slab is calculated using the Maxwell-Garnett Effective Media model of
    each of the constituent Scatterers:

    >>> np.sum([complex(sld) * vf / np.sum(vf_list) for sld, vf in
    ...         zip(sld_list, vf_list)]).

    See Reference:
    Vadim A. Markel, "Introduction to the Maxwell Garnett
    approximation: tutorial," J. Opt. Soc. Am. A 33, 1244-1256 (2016)
    """

    def __init__(
        self,
        thick,
        rough,
        sld_list,
        vf,
        shapefunc = [1/3, 1/3, 1/3],
        name="",
    ):
        super(PXR_MGMixedSlab, self).__init__(name=name)
        
        self.thick = possibly_create_parameter(thick, name="%s - thick" % name)
        self.vf = possibly_create_parameter((vf/100 if vf>1 else vf), name=f"vf - {name}", bounds=(0.0, 1.0))
        self.rough = possibly_create_parameter(rough, name=f"{name} - rough")

        self.sld = []
        self._sld_parameters = Parameters(name=f"{name} - slds")
        self._vf_parameters = Parameters(name=f"{name} - volfracs")

        i = 0
        for s in sld_list:
            if isinstance(s, PXR_Scatterer):
                self.sld.append(s)
            else:
                self.sld.append(PXR_SLD(s))
            self._sld_parameters.append(self.sld[-1].parameters)
            i += 1

        p = Parameters(name=self.name)
        p.append([self.thick, self.vf])
        p.extend(self._sld_parameters)

        self._parameters = p
        self.shapefunc = np.array(shapefunc)

    def __repr__(self):
        return (
            f"PXR_MGMixedSlab({self.thick!r}, {self.sld!r}, {self.vf!r}, {self.shapefunc!r},"
            f" {self.rough!r}, name={self.name!r},"
        )

    def __str__(self):
        return str(self.parameters)

    @property
    def parameters(self):
        """
        :class:`refnx.analysis.Parameters` associated with this component

        """
        self._parameters.name = self.name
        return self._parameters

    def slabs(self, structure=None):
        """
        Slab representation of this component. See :class:`Component.slabs`
        """

        MGdelta = np.trace(self.tensor().real)/3
        MGbeta = np.trace(self.tensor().imag)/3
        sldc = complex(MGdelta, MGbeta)
        
        return np.array(
            [
                [
                    self.thick.value,
                    sldc.real,
                    sldc.imag,
                    self.rough.value,
                ]
            ]
        )
        
    def tensor(self, structure=None,energy=None):
        """
        Stored information pertaining to the tensor dielectric properties of the slab.
        Value of the tensor can be updated by changing the energy encoded into the SLD object.
        self.sld.energy = ()
        The trace of the layer is stored in the .slabs() attribute as the real and imaginary component of the SLD
        """
        vf = self.vf.value
        host = np.diag(self.sld[0].tensor) #Just grab the diagonal elements for easy calculation
        dopant = np.diag(self.sld[1].tensor) #Just grab the diagonal elements for easy calculation
        
        if (energy is not None and hasattr(self.sld, 'energy')):
            self.sld.energy = energy
        
        numerator = host + (self.shapefunc * (1-vf) + (vf))*(dopant - host)
        denom = host + self.shapefunc * (1-vf) * (dopant - host)
        
        combinetensor = host * (numerator/denom)
        
        return combinetensor * np.eye(3)#self.sld.tensor 
        
#Support Functions


        """
        Calculates an index of refraction profile. as a function of distance from superstrate.

        Parameters
        ----------
        
        z : float
            Interfacial distance (Angstrom) measured from interface between the
            fronting medium and the first layer.
        align: int, optional
            Places a specified interface in the slab representation of a
            Structure at z = 0. Python indexing is allowed, e.g. supplying -1
            will place the backing medium at z = 0.

        Returns
        -------
        
        zed : np.ndarray
            Interfacial distance measured from superstrate offset by 'align'
            Has shape (N, ) 
            N - number of data points
        prof : np.ndarray / complex
            Real and imaginary tensor components of index of refraction / unitless
            Has shape (N, 3)
            - prof[N, 0]
                dielectric component xx at depth N
            - prof[N, 1]
                dielectric component yy at depth N
            - prof[N, 2]
                dielectric component zz at depth N

        Notes
        -----
        This can be called in vectorised fashion.
        
        """