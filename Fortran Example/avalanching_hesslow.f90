! avalanching_test.f90
! constants 
module physconst
  implicit none
  integer, parameter :: dp = kind(1.0d0)
  real(dp), parameter :: e = 1.602176634d-19
  real(dp), parameter :: epsilon0 = 8.8541878128d-12
  real(dp), parameter :: me = 9.1093837015d-31
  real(dp), parameter :: c = 2.99792458d8
  real(dp), parameter :: pi = 3.14159265358979323846d0
end module physconst


module delta_funcs
  use physconst
  implicit none
contains

  function calcDelta(Eceff,nuD0,nuS1,tauRadInv,phib1,phib2,pc0) result(delta)
    real(dp), intent(in) :: Eceff,nuD0,nuS1,tauRadInv,phib1,phib2,pc0
    real(dp) :: delta
    delta = (nuD0/(nuS1**2)) * (nuD0*tauRadInv/Eceff + phib1 + phib2*log(pc0))
  end function calcDelta

  function calcEceff(delta,nuS0,nuS1,nuD1,nuD0,pc0) result(Eceff)
    real(dp), intent(in) :: delta,nuS0,nuS1,nuD1,nuD0,pc0
    real(dp) :: Eceff
    Eceff = nuS0 + nuS1 * ( (1.0_dp + nuD1/nuD0)*log(pc0) + sqrt(2.0_dp*delta + 1.0_dp) )
  end function calcEceff

end module delta_funcs


module ec_module
  use physconst
  use delta_funcs
  implicit none
contains

  subroutine getEc_eff(Z,Z0,nj,nSpecies,T,B,element,Eceff_out,Ectot_out)
    integer, intent(in) :: nSpecies
    real(dp), intent(in) :: Z(nSpecies), Z0(nSpecies), nj(nSpecies)
    real(dp), intent(in) :: T, B
    character(len=*), intent(in) :: element
    real(dp), intent(out) :: Eceff_out, Ectot_out

    real(dp), allocatable :: ln_aBar(:), ln_I(:)
    real(dp) :: ne_free, ntot, Zeff, Zfulleff, lnLStar
    real(dp) :: nuD0, nuD1, nuS0, nuS1
    real(dp), allocatable :: Ne(:)
    real(dp) :: tauRadInv, ALPHA, bremsprefactor, phib1, phib2
    real(dp) :: pc0, Ectot, Eceff0, delta0, delta1, EceffOverEctot
    integer :: i

    ! pick the element
    if (trim(element) == "Ne") then
        allocate(ln_aBar(10), ln_I(10))
        ln_aBar = (/4.7064_dp,4.6097_dp,4.5028_dp,4.3858_dp,4.2638_dp,4.1266_dp,3.9568_dp,3.6810_dp, &
                    3.1777_dp,3.1278_dp/)
        ln_I    = (/-8.2227_dp,-8.0370_dp,-7.8614_dp,-7.6837_dp,-7.4994_dp,-7.2788_dp, &
                     -6.9808_dp,-6.5976_dp,-5.8933_dp,-5.8320_dp/)
    else if (trim(element) == "Ar") then
        allocate(ln_aBar(18), ln_I(18))
        ln_aBar = (/4.5677_dp,4.5007_dp,4.4289_dp,4.3513_dp,4.2698_dp,4.1815_dp,4.0829_dp,3.9748_dp, &
                    3.8511_dp,3.7882_dp,3.7203_dp,3.6436_dp,3.5579_dp,3.4525_dp,3.3093_dp,3.0503_dp, &
                    2.5578_dp,2.5332_dp/)
        ln_I    = (/-7.9050_dp,-7.7532_dp,-7.6076_dp,-7.4626_dp,-7.3178_dp,-7.1665_dp,-7.0055_dp, &
                    -6.8020_dp,-6.5538_dp,-6.4647_dp,-6.3644_dp,-6.2465_dp,-6.1070_dp,-5.9219_dp, &
                    -5.6535_dp,-5.3213_dp,-4.6937_dp,-4.6598_dp/)
    else
        print *, "ERROR: element must be 'Ne' or 'Ar'"
        stop
    end if

    allocate(Ne(nSpecies))

    ! free and total electron densities
    Ne = Z - Z0
    ne_free = sum(Z0*nj)
    ntot    = sum(Z*nj)

    Zeff = sum((Z0**2)*nj) / ne_free
    Zfulleff = sum((Z**2)*nj) / ne_free

    lnLStar = 14.6_dp + 0.5_dp * log( T / (ne_free/1.0e20_dp) )

    ! compute nu constants
    nuD0 = (1.0_dp + Zeff) - (2.0_dp/3.0_dp) * sum( (Ne**2)*nj )/ne_free / lnLStar
    nuD1 = Zfulleff / lnLStar
    nuS0 = 1.0_dp
    nuS1 = (1.0_dp + 3.0_dp*sum(Ne*nj)/ne_free) / (2.0_dp*lnLStar)

    do i = 2, nSpecies
        nuD0 = nuD0 + (Z(i)**2 - Z0(i)**2) * ln_aBar(int(Z0(i))+1) * (nj(i)/ne_free) / lnLStar
        nuS0 = nuS0 + Ne(i) * (-ln_I(int(Z0(i))+1) - 1.0_dp) * (nj(i)/ne_free) / lnLStar
    end do

    ! radiation terms
    tauRadInv = B**2 / (ne_free/1e20_dp) / (15.44_dp * lnLStar)

    ALPHA = 1.0_dp/137.036_dp
    bremsprefactor = ALPHA * Zfulleff / lnLStar
    phib1 = bremsprefactor * 0.35_dp
    phib2 = bremsprefactor * 0.20_dp

    pc0 = nuD0/(2.0_dp*nuS1)

    ! iterate calculation
    Ectot = sum(Z*nj)/ne_free
    Eceff0 = Ectot
    delta0 = calcDelta(Eceff0,nuD0,nuS1,tauRadInv,phib1,phib2,pc0)
    delta1 = calcDelta(calcEceff(delta0,nuS0,nuS1,nuD1,nuD0,pc0), &
                       nuD0,nuS1,tauRadInv,phib1,phib2,pc0)

    EceffOverEctot = calcEceff(delta1,nuS0,nuS1,nuD1,nuD0,pc0)/Ectot

    ! convert Ectot to SI
    Ectot = sum(Z*nj) * e**3 * lnLStar /(4.0_dp*pi*epsilon0**2 * me * c**2)

    Eceff_out = Ectot * EceffOverEctot
    Ectot_out = Ectot

    if (allocated(ln_aBar)) deallocate(ln_aBar)
    if (allocated(ln_I))    deallocate(ln_I)
    if (allocated(Ne))      deallocate(Ne)

  end subroutine getEc_eff

end module ec_module


module nu_module
  use physconst
  implicit none
contains

  subroutine get_nuD_nuS(p,Z,Z0,nj,nSpecies,T,element,nuD,nuS)
    real(dp), intent(in) :: p, T
    real(dp), intent(in) :: Z(nSpecies), Z0(nSpecies), nj(nSpecies)
    integer, intent(in) :: nSpecies
    character(len=*), intent(in) :: element
    real(dp), intent(out) :: nuD, nuS

    real(dp), allocatable :: ln_aBar(:), ln_I(:)
    real(dp), allocatable :: aBar_Ne(:), I_Ne(:)
    real(dp) :: ne_free, Zeff
    real(dp), allocatable :: Ne(:)
    real(dp) :: gamma_rel, beta
    real(dp) :: PT, lnAc, lnA0, lnAee, lnAei
    real(dp) :: hj, F1j, pabar1p5
    integer :: j
    real(dp), parameter :: k = 5.0_dp

    if (trim(element) == "Ne") then
        allocate(ln_aBar(10), ln_I(10))
        ln_aBar = (/4.7064_dp,4.6097_dp,4.5028_dp,4.3858_dp,4.2638_dp,4.1266_dp,3.9568_dp,3.6810_dp, &
                    3.1777_dp,3.1278_dp/)
        ln_I    = (/-8.2227_dp,-8.0370_dp,-7.8614_dp,-7.6837_dp,-7.4994_dp,-7.2788_dp, &
                    -6.9808_dp,-6.5976_dp,-5.8933_dp,-5.8320_dp/)
    else if (trim(element) == "Ar") then
        allocate(ln_aBar(18), ln_I(18))
        ln_aBar = (/4.5677_dp,4.5007_dp,4.4289_dp,4.3513_dp,4.2698_dp,4.1815_dp,4.0829_dp,3.9748_dp, &
                    3.8511_dp,3.7882_dp,3.7203_dp,3.6436_dp,3.5579_dp,3.4525_dp,3.3093_dp,3.0503_dp, &
                    2.5578_dp,2.5332_dp/)
        ln_I    = (/-7.9050_dp,-7.7532_dp,-7.6076_dp,-7.4626_dp,-7.3178_dp,-7.1665_dp,-7.0055_dp, &
                    -6.8020_dp,-6.5538_dp,-6.4647_dp,-6.3644_dp,-6.2465_dp,-6.1070_dp,-5.9219_dp, &
                    -5.6535_dp,-5.3213_dp,-4.6937_dp,-4.6598_dp/)
    else
        print *, "ERROR: element must be 'Ne' or 'Ar'"
        stop
    end if

    allocate(aBar_Ne(size(ln_aBar)), I_Ne(size(ln_I)))
    aBar_Ne = exp(ln_aBar)
    I_Ne    = exp(ln_I)

    allocate(Ne(nSpecies))
    Ne = Z - Z0
    ne_free = sum(Z0*nj)
    Zeff = sum((Z0**2)*nj) / ne_free

    gamma_rel = sqrt(1.0_dp + p*p)
    beta = p*p / (1.0_dp + p*p)

    PT = sqrt( 2.0_dp * T / (me * c*c) )

    lnAc = 14.6_dp + 0.5_dp * log( T / (ne_free/1e20_dp) )
    lnA0 = 14.9_dp - 0.5_dp*log(ne_free/1e20_dp) + log(T/1e3_dp)

    lnAee = lnA0 + 0.2_dp * log( 1.0_dp + ( 2.0_dp*(gamma_rel-1.0_dp)/PT**2 )**2.5_dp )
    lnAei = lnA0 + 0.2_dp * log( 1.0_dp + ( 2.0_dp*p/PT )**5 )

    nuS = lnAee
    nuD = lnAee + Zeff * lnAei

    do j = 2, nSpecies      ! matches python: skip index 0
        hj = p * sqrt(gamma_rel-1.0_dp) / I_Ne(int(Z0(j))+1)

        F1j = (1.0_dp/k) * log(1.0_dp + hj**k) - beta**2

        pabar1p5 = (p * aBar_Ne(int(Z0(j))+1))**1.5_dp

        nuS = nuS + (nj(j)/ne_free) * Ne(j) * F1j

        nuD = nuD + (nj(j)/ne_free) * ( &
             (2.0_dp/3.0_dp)*(Z(j)**2 - Z0(j)**2) * log(pabar1p5 + 1.0_dp)   &
           - (2.0_dp/3.0_dp)*(Ne(j)**2) * (pabar1p5/(pabar1p5+1.0_dp)) )
    end do

    nuS = nuS / lnAc
    nuD = nuD / lnAc

    if (allocated(ln_aBar)) deallocate(ln_aBar)
    if (allocated(ln_I))    deallocate(ln_I)
    if (allocated(aBar_Ne)) deallocate(aBar_Ne)
    if (allocated(I_Ne))    deallocate(I_Ne)
    if (allocated(Ne))      deallocate(Ne)

  end subroutine get_nuD_nuS


  function fn(pstar, Estar, Z, Z0, nj, nSpecies, T, element) result(val)
    real(dp), intent(in) :: pstar, Estar, T
    real(dp), intent(in) :: Z(nSpecies), Z0(nSpecies), nj(nSpecies)
    integer, intent(in) :: nSpecies
    character(len=*), intent(in) :: element
    real(dp) :: val, nuD, nuS

    call get_nuD_nuS(pstar, Z, Z0, nj, nSpecies, T, element, nuD, nuS)

    val = pstar * sqrt(Estar) - (nuS * nuD)**0.25_dp
  end function fn

end module nu_module


module hesslow_module
  use physconst
  use ec_module
  use nu_module
  implicit none
contains

  function getConnorHastieField(ne,lnA) result(Ec)
    real(dp), intent(in) :: ne, lnA
    real(dp) :: Ec
    Ec = e**3 * ne * lnA / (me * c**2 * 4.0_dp * pi * epsilon0**2)
  end function getConnorHastieField

! Probably check here
  function get_pstar(E_star, Z, Z0, nj, nSpecies, T, element) result(pstar)
    real(dp), intent(in) :: E_star, T
    real(dp), intent(in) :: Z(nSpecies), Z0(nSpecies), nj(nSpecies)
    integer, intent(in) :: nSpecies
    character(len=*), intent(in) :: element
    real(dp) :: pstar

    real(dp), parameter :: x0 = 1.0_dp
    real(dp), parameter :: eps = 1.0e-3_dp
    real(dp), parameter :: thres = 1.0e-4_dp
    integer, parameter  :: nmax = 100

    integer :: ii
    real(dp) :: x, xnew, f_x, fp_x, res

    x = x0
    res = 1.0e3_dp
    ii  = 0

    do while (ii < nmax .and. res > thres)
        f_x = fn(x, E_star, Z, Z0, nj, nSpecies, T, element)
        fp_x = ( fn(x+eps, E_star, Z, Z0, nj, nSpecies, T, element) &
               - fn(x-eps, E_star, Z, Z0, nj, nSpecies, T, element) ) / (2.0_dp*eps)
        if (fp_x == 0.0_dp) then
            exit
        end if
        xnew = x - f_x / fp_x
        if (x /= 0.0_dp) then
            res  = abs( (xnew - x)/x )
        else
            res = abs(xnew - x)
        end if
        x = xnew
        ii = ii + 1
    end do

    pstar = x
  end function get_pstar


  function getHesslowAvalanchingRate(E_star,Z,Z0,nj,nSpecies,T,element,B) result(gamma)
    real(dp), intent(in) :: E_star, T, B
    real(dp), intent(in) :: Z(nSpecies), Z0(nSpecies), nj(nSpecies)
    integer, intent(in) :: nSpecies
    character(len=*), intent(in) :: element
    real(dp) :: gamma

    real(dp) :: ne_free, ntot, lnAc
    real(dp) :: Ec, Epar, Eceff
    real(dp) :: pstar
    real(dp) :: nuD, nuS, Zeff
    real(dp) :: Ectmp

    ne_free = sum(Z0 * nj)
    ntot    = sum(Z  * nj)

    lnAc = 14.6_dp + 0.5_dp * log( T / (ne_free/1e20_dp) )

    Ec = getConnorHastieField(ne_free, lnAc)
    Epar = Ec * E_star

    if (abs(ne_free - ntot) < 1.0e-12_dp) then
        Zeff = sum((Z0**2)*nj) / ne_free
        Eceff = Ec
        nuD = 1.0_dp + Zeff
        nuS = 1.0_dp
    else
        call getEc_eff(Z,Z0,nj,nSpecies,T,B,element,Eceff,Ectmp)
        pstar = get_pstar(E_star, Z, Z0, nj, nSpecies, T, element)
        call get_nuD_nuS(pstar, Z, Z0, nj, nSpecies, T, element, nuD, nuS)
    end if

    gamma = ( e / (me*c*lnAc) ) * (ntot/ne_free) * (Epar - Eceff) / sqrt(4.0_dp + nuS*nuD)

  end function getHesslowAvalanchingRate

end module hesslow_module




