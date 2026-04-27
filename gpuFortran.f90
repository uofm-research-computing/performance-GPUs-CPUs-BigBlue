program gpu2
  use iso_fortran_env, only : real32
  use OMP_LIB
  implicit none

  type :: neigh
     real(real32) :: n, s, e, w
  end type neigh
  integer, parameter :: MAX_ITER = 1000
  real(real32), parameter :: THRESHOLD = 0.001_real32

  integer :: nx, ny, statesize
  integer, allocatable, dimension(:) :: seed
  integer :: i, j, iter
  character(len=12),dimension(:), allocatable :: args
  real(real32), allocatable :: field(:,:), nextfield(:,:)
  real(real32) :: error, derror
  type(neigh) :: boundary
  type(neigh) :: nbr

  integer :: arg_count

  arg_count = command_argument_count()
  if (arg_count /= 3) then
     print *, "Usage: gpu2 nx ny seed"
     print *, "Number of available devices:", omp_get_num_devices()
     stop
  end if
  allocate(args(arg_count))
  do i=1,arg_count
     call get_command_argument(i, args(i))
  end do
  read(args(1),*) nx
  read(args(2),*) ny
  call random_seed( size=statesize )
  allocate(seed(statesize))
  call random_seed( get=seed )
  read(args(3),*) seed(1)
  call random_seed(put=seed)

  allocate(field(nx, ny), nextfield(nx, ny))

  call random_number(field)

  boundary%n = 1.0
  boundary%s = 1.0
  boundary%e = 0.0
  boundary%w = 0.0

  iter = 0

  !$omp target enter data map(to: field, nextfield, boundary)

  do
     !$omp target teams distribute parallel do private(nbr)
     do j = 1, ny
        do i = 1, nx
           if (j < ny) then
              nbr%n = field(i, j+1)
           else
              nbr%n = boundary%n
           end if
           
           if (j > 1) then
              nbr%s = field(i, j-1)
           else
              nbr%s = boundary%s
           end if
           
           if (i < nx) then
              nbr%e = field(i+1, j)
           else
              nbr%e = boundary%e
           end if
           
           if (i > 1) then
              nbr%w = field(i-1, j)
           else
              nbr%w = boundary%w
           end if
           nextfield(i,j) = 0.25_real32 * (nbr%n + nbr%s + nbr%e + nbr%w)
        end do
     end do

     error = 0.0_real32

     !$omp target teams distribute parallel do reduction(max:error)
     do j = 1, ny
        do i = 1, nx
           derror = abs(nextfield(i,j) - field(i,j))
           error = max(error, derror)
           field(i,j) = nextfield(i,j)
        end do
     end do

     iter = iter + 1
     if (error <= THRESHOLD .or. iter >= MAX_ITER) exit
  end do

  !$omp target exit data map(from: field)

  print *, "Converged after", iter, "iterations with maximum error", error

  deallocate(field, nextfield)

end program gpu2

