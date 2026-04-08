program make_hot_tail_figure
    use physconst
    use HT_funcs
    implicit none

    integer, parameter :: N = 8000

    real(dp) :: Tinit, Tfinal, t_char, ne
    real(dp) :: time, Temp, vT0, vT
    real(dp) :: nu0, tau
    real(dp) :: x(N), v(N)
    real(dp) :: fm0(N), fm1(N), fm2(N)
    real(dp) :: f1(N), f2(N)
    real(dp) :: norm_fac
    real(dp) :: dx
    integer :: i

    ! parameters
    Tinit = 3.1d3
    Tfinal = 0.01d0*Tinit
    t_char = 5.d-4
    ne = 2.8d19

    vT0 = sqrt(2.d0*Tinit*e/me)

    print *, "vT0:", vT0

    ! velocity grid
    dx = 4.d0/real(N-1,dp)

    do i=1,N
        x(i) = dx*real(i-1,dp)
        v(i) = sqrt(x(i))*vT0
    end do

    ! ---- t = 0 Maxwellian ----
    do i=1,N
        fm0(i) = getfm(v(i),ne,vT0)
    end do

    norm_fac = maxval(fm0)

    ! ---- t = 0.75 ms ----
    time = 7.5d-4

    Temp = temperature(time,Tinit,Tfinal,t_char)
    vT = sqrt(2.d0*Temp*e/me)

    do i=1,N
        fm1(i) = getfm(v(i),ne,vT)
    end do

    nu0 = getnu0(ne,Tinit)
    tau = getTau(nu0,time,t_char)

    do i=1,N
        f1(i) = getf(v(i),ne,vT0,tau)
    end do

    ! ---- t = 1.5 ms ----
    time = 1.5d-3

    Temp = temperature(time,Tinit,Tfinal,t_char)
    vT = sqrt(2.d0*Temp*e/me)

    do i=1,N
        fm2(i) = getfm(v(i),ne,vT)
    end do

    nu0 = getnu0(ne,Tinit)
    tau = getTau(nu0,time,t_char)

    do i=1,N
        f2(i) = getf(v(i),ne,vT0,tau)
    end do

    ! ---- write CSV ----
    open(unit=10,file="figure_4b.csv",status="replace")

    write(10,*) "x,fm0,fm1,f1,fm2,f2"

    do i=1,N
        write(10,'(E18.10,",",E18.10,",",E18.10,",",E18.10,",",E18.10,",",E18.10)') &
        x(i), &
        fm0(i)/norm_fac, &
        fm1(i)/norm_fac, &
        f1(i)/norm_fac, &
        fm2(i)/norm_fac, &
        f2(i)/norm_fac
    end do

    close(10)

    print *, "CSV file written: figure_4b.csv"

end program make_hot_tail_figure