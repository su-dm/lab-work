import math
def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
    # det(A-LI) = 0
    # A - LI = [a-L, b],[c,d-L]
    # det is ad - bc => (a-L)(d-L) - bc
    #ad-aL-dL+L^2-bc=0
    #factor L^2 -(a+d)L +(ad-bc)=0
    a = matrix[0][0]
    b = matrix[0][1]
    c = matrix[1][0]
    d = matrix[1][1]

    #quadratic formula (-b +/- sqrt(b^2-4ac))/2a
    # denote the quadratic formula ones with q
    aq = 1
    bq = -(a+d)
    cq = (a*d) - (b*c)
    
    discriminant = bq**2-(4*aq*cq)
    # complex eigen value
    if discriminant < 0:
        import cmath #imag number
        sqrt_disc = cmath.sqrt(discriminant)
    else:
        sqrt_disc = math.sqrt(discriminant)
    e1 = (-bq + sqrt_disc)/(2*aq)
    e2 = (-bq - sqrt_disc)/(2*aq) 
    eigenvalues = [e1, e2]
	return eigenvalues
