program activated_sources

print *, 'Running activated_sources.f90'
! DECLARE VARIABLES

DO i=-1,11
        Estar = 2.0**i
        print *, 'Estar, Wc [keV], Beta Rate [per s], Comp. Rate [per s] =',Estar, & 
                get_Wc(Estar)/1e3,beta_source(Estar),compton_source(Estar) 
END DO

end 


function get_Wc(E_star) result(outval)
          ! Returns the critical RE energy in [eV]
          ! E_star = Par. E-field normalized with CH electric field
          real, intent(in) :: E_star
          real :: outval
          real alpha
          IF (E_star <= 1) THEN
                  alpha = 1.0 + 1.0e-6 ! Set to min. value for calculation
          ELSE
                 alpha = 1.0 * E_star
          ENDIF
          outval = 9.1093837015e-31 * (3.0e8)**2 * ((1-1.0/alpha)**(-0.5)-1) / 1.602e-19 ! Critical energy in [eV]
end

function beta_source(E_star) result(outval)
		! Returns the beta source rate in per second
		! Multiply with tritium denisty to get generation rate in per
		! cubic m
		! per second
		real, intent(in) :: E_star  !E-field normalized by Connor-hastie field
		real :: outval
		real A, mu, sig ! Fit parameters
		real Wc ! Critical Runaway energy in keV
		Wc = get_Wc(E_star) * 1e-3 ! keV
		A =2.09023917e-09
		mu = -3.80424810
		sig = 6.81315848
		
		IF (Wc > 18.6) THEN
				outval = 0.0
		ELSEIF (Wc < 0.1) THEN
				outval = 1.8e-9 ! Maximum Rate
		ELSE
				outval = A * exp(-(Wc-mu)**2/(2*sig**2)) 
		ENDIF

end

function compton_source(E_star) result(outval)
		! Returns compton rate in per second
		! Multiply with total electron denisty to get rate in per cubic
		! m per
		! second
		real, intent(in) :: E_star ! E-field normalized by Connor-hastie field
		real :: outval
		real Wc, x
		real A, x0, L, c, B ! Fitting Parameters
		
		Wc = get_Wc(E_star) * 1e-3 ! Critical Runaway energy in [keV]
		x = log(Wc)
		IF (Wc < 0.1) THEN
				outval = 4.3e-11 ! MAX rate for SPARC
		ELSEIF (Wc > 10**4.5) THEN
				outval = 0.0
		ELSE
				! Fit parameters for SPARC
				A = 6.00578227e-01
				L = 1.25488546e+00
				x0 = 4.44639341e+00
				c = 2.12986727e-11
				B = -2.17893889e-11
				outval = B * tanh(A *(x-x0)/L) + c
		ENDIF
end
