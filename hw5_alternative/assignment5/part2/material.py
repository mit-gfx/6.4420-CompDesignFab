from abc import ABC, abstractmethod
from numpy import ndarray as array

import numpy as np


class Material(ABC):
    '''
    Abstract class for material models.
    '''
    def __init__(self, E: float, nu: float):
        '''
        E - Young's modulus, nu - Poisson's ratio.
        '''
        # Save the base parameters
        self.E = E
        self.nu = nu

        # Compute the Lame parameters
        self.mu = E / (2 * (1 + nu))
        self.lm = E * nu / ((1 + nu) * (1 - 2 * nu))

        self.type = 'unknown'

    @abstractmethod
    def energy_density(self, F: array) -> float: ...

    @abstractmethod
    def stress_tensor(self, F: array) -> array: ...

    @abstractmethod
    def stress_differential(self, F: array) -> array: ...


class LinearElastic(Material):
    '''
    Implements the linear elastic material.
    '''
    def __init__(self, E: float, nu: float):
        # Simply invoke the constructor of the parent class
        super().__init__(E, nu)

        self.type = 'linear'

    def energy_density(self, F: array) -> float:
        '''
        Compute the energy density W.
        Formula:
            eps = 0.5 * (F + F.T) - I
            W = mu * (eps : eps) + 0.5 * lm * tr(eps) ** 2
        '''
        strain = 0.5 * (F + F.T) - np.eye(F.shape[0])
        strain_contract = np.dot(strain.ravel(), strain.ravel())
        W = self.mu * strain_contract + 0.5 * self.lm * strain.trace() ** 2
        return W

    def stress_tensor(self, F: array) -> array:
        '''
        Compute the stress tensor P.
        Formula:
            P = mu * (F + F.T - 2I) + lm * (tr(F) - dim) * I
        '''
        dim, I = F.shape[0], np.eye(F.shape[0])
        P = self.mu * (F + F.T - 2 * I) + self.lm * (F.trace() - dim) * I
        return P

    def stress_differential(self, F: array) -> array:
        '''
        Compute the differential of the stress tensor P w.r.t. the deformation gradient F.

        Params:
            * `F: array` - (dxd) the deformation gradient, d = #dimensions

        Return value:
            * `dP_dF: array` - (d^2 x d^2) the gradient of the stress tensor P w.r.t. F
        '''
        # Here we use the chain rule to compute dP/dF.
        #   - First, we compute D1 = d(F + F.T)/dF, which equals to I + d(F.T)/dF
        #   - Then, we compute D2 = d(F.trace() * I)/dF
        #   - Finally, we combine D1 and D2 to obtain dP/dF
        # You will complete this process in the code below

        # Constants
        dim, dim2 = F.shape[0], F.shape[0] ** 2
        mu, lm = self.mu, self.lm

        # Compute d(F.T)/dF
        # --------
        # TODO: Your code here. Think about which elements in d(F.T)/dF are non-zero.
        dFT_dF = np.zeros((dim2, dim2))
        for i in []:        # <--
            for j in []:    # <--
                ...         # <--

        # Compute D1
        # --------
        # TODO: Your code here.
        # HINT: The `np.eye(n)` function creates an identify matrix of size nxn
        D1 = np.zeros((dim2, dim2))     # <--

        # Compute D2 = d(F.trace() * I)/dF
        # --------
        # TODO: Your code here. Think about which elements in D2 are non-zero
        D2 = np.zeros((dim2, dim2))
        for i in []:        # <--
            for j in []:    # <--
                ...         # <--

        # Compute dP/dF
        # --------
        # TODO: Your code here.
        dP_dF = np.zeros((dim2, dim2))      # <--

        return dP_dF


class NeoHookean(Material):
    '''
    Implements the Neo-Hookean material model.
    '''
    def __init__(self, E: float, nu: float):
        # Simply invoke the constructor of the parent class
        super().__init__(E, nu)

        self.type = 'nonlinear'

    def energy_density(self, F: array) -> float:
        '''
        Compute the energy density W.
        Formula:
            I1 = tr(F.T * F)
            J = det(F)
            W = 0.5 * mu * (I1 - dim - 2 * log(J)) + 0.5 * lm * log(J) ** 2
        '''
        dim = F.shape[0]
        I1 = np.dot(F.ravel(), F.ravel())
        logJ = np.log(np.linalg.det(F))
        W = 0.5 * self.mu * (I1 - dim - 2 * logJ) + 0.5 * self.lm * logJ ** 2
        return W

    def stress_tensor(self, F: array) -> array:
        '''
        Compute the stress tensor P.
        Formula:
            P = mu * (F - F^(-T)) + lm * log(J) * F^(-T)
        '''
        F_invT = np.linalg.inv(F).T
        logJ = np.log(np.linalg.det(F))
        P = self.mu * (F - F_invT) + self.lm * logJ * F_invT
        return P

    def stress_differential(self, F: array) -> array:
        '''
        Compute the differential of the stress tensor P w.r.t. the deformation gradient F.

        Params:
            * `F: array` - (dxd) the deformation gradient, d = #dimensions

        Return value:
            * `dP_dF: array` - (d^2 x d^2) the gradient of the stress tensor P w.r.t. F
        '''
        # In this case, dP/dF is much more complicated
        #   - Compute D1 = mu * I
        #   - Compute D2 = lm * vec(F^(-T)) * vec(F^(-T))^T
        #   - Compute D3 = (lm * log(J) - mu) * d(F^(-T))/dF
        #     where d(F^(-T))/dF = - F^(-T) * d(F^T)/dF * F^(-T)

        # Constants
        dim, dim2 = F.shape[0], F.shape[0] ** 2
        mu, lm = self.mu, self.lm

        # Compute D1
        D1 = mu * np.eye(dim2)

        # Compute D2
        # F_invT flattened by columns is equivalent to F_inv flattened by rows
        F_invT_vec = np.linalg.inv(F).ravel()
        F_invT_outer = np.outer(F_invT_vec, F_invT_vec)
        D2 = lm * F_invT_outer

        # Compute D3
        # Understanding the transpose is the hardest part
        # Let (i, j) be the two dimensions of F_invT, and the same for (k, s). The axis order in
        # F_invT_outer is (j, i, s, k) since the matrix is flattened column by column.
        # However, the axis order of d(F^(-T))/dF is (s, i, j, k) according to the right hand side.
        # Thus, the transpose is from (j, i, s, k) to (s, i, j, k), organized as (2, 1, 0, 3).
        coeff = lm * np.log(np.linalg.det(F)) - mu
        D3 = -coeff * F_invT_outer.reshape(dim, dim, dim, dim)
        D3 = D3.transpose(2, 1, 0, 3).reshape(dim2, dim2)

        # Compute dP/dF
        dP_dF = D1 + D2 + D3
        return dP_dF
