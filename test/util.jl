module TestUtil

using CrystalTree: find_first_unassigned, cos_angle

using Test

@test cos_angle([1,2,3], [1,2,3]) == 1
# @test cosangle


end # module