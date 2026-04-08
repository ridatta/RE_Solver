program test_ht_rate
    use physconst
    use HT_funcs
    implicit none

    integer, parameter :: N = 200

    real(dp) :: T0, Tf, t_char, n0, E0
    real(dp) :: t(N), nht(N), rate(N)
    real(dp) :: Temp
    real(dp) :: tmin, tmax, dt
    real(dp) :: nht_new
    integer :: i

    ! parameters
    T0     = 3.1d3
    Tf     = 0.01d0*T0
    t_char = 3.d-4
    n0     = 2.8d19
    E0     = 0.01d0

    ! time grid
    tmin = 0.d0
    tmax = 2.d-3
    dt = (tmax - tmin)/real(N-1,dp)

    do i = 1, N
        t(i) = tmin + dt*real(i-1,dp)
    end do

    nht  = 0.d0
    rate = 0.d0

    do i = 2, N

        Temp = temperature(t(i), T0, Tf, t_char)

        rate(i) = get_HT_rate( &
                    ne=n0, &
                    temp=Temp, &
                    t=t(i), &
                    T0=T0, &
                    E0=E0, &
                    nht=nht(i-1), &
                    dt=dt, &
                    nht_new=nht_new, &
                    td=t_char)

        nht(i) = nht_new

    end do


    ! write CSV
    open(unit=10,file="HT_rate_scan.csv",status="replace")

    write(10,*) "t_ms,HT_rate_m3_per_ms,nht"

    do i = 1, N
        write(10,'(E18.10,",",E18.10,",",E18.10)') &
            t(i)*1.d3, &
            rate(i)/1.d3, &
            nht(i)
    end do

    close(10)

    print *, "File written: HT_rate_scan.csv"

end program test_ht_rate