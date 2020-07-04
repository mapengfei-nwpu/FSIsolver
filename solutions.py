
# 零Dirichlet边界条件
# 区域都是[0,1]x[0][0,1], T=0.1, delta=0.1

solutions = []
# 1.1
solution = {}
ux = "-exp(t) * x[0]*x[0] * (x[0] - 1)*(x[0] - 1) * x[1] * (x[1] - 1) * (2 * x[1] - 1) / 256"
uy = "exp(t) * x[0] * (x[0] - 1) * (2 * x[0] - 1) * x[1]*x[1] * (x[1] - 1)*(x[1] - 1)/256"
p = "exp(t) * (x[0]*x[0]*x[0] - 1/4)"
fx = "3*x[0]*x[0]*exp(t) + (x[0]*x[0]*exp(t)*(x[0] - 1)*(x[0] - 1)*(x[1] - 1))/6400 + (x[0]*x[0]*exp(t)*(2*x[1] - 1)*(x[0] - 1)*(x[0] - 1))/12800 + (x[0]*x[0]*x[1]*exp(t)*(x[0] - 1)*(x[0] - 1))/6400 + (x[0]*x[0]*x[1]*exp(t)*(2*x[1] - 1)*(x[1] - 1))/12800 + (x[1]*exp(t)*(2*x[1] - 1)*(x[0] - 1)*(x[0] - 1)*(x[1] - 1))/12800 + (x[0]*x[1]*exp(t)*(2*x[0] - 2)*(2*x[1] - 1)*(x[1] - 1))/6400 - (x[0]*x[0]*x[1]*exp(t)*(2*x[1] - 1)*(x[0] - 1)*(x[0] - 1)*(x[1] - 1))/256 - (x[0]*x[0]*x[0]*x[1]*x[1]*exp(2*t)*(2*x[0] - 1)*(x[0] - 1)*(x[0] - 1)*(x[0] - 1)*(x[1] - 1)*(x[1] - 1)*(6*x[1]*x[1] - 6*x[1] + 1))/65536 + (x[0]*x[0]*x[0]*x[1]*x[1]*exp(2*t)*(2*x[1] - 1)*(x[0] - 1)*(x[0] - 1)*(x[1] - 1)*(2*x[0]*x[0] - 3*x[0] + 1)*(2*x[1]*x[1] - 3*x[1] + 1))/32768"
fy = "(x[0]*x[1]*x[1]*exp(t)*(2*x[0] - 1)*(x[0] - 1)*(x[1] - 1)*(x[1] - 1))/256 - (x[1]*x[1]*exp(t)*(2*x[0] - 1)*(x[1] - 1)*(x[1] - 1))/12800 - (x[0]*x[1]*x[1]*exp(t)*(x[1] - 1)*(x[1] - 1))/6400 - (x[0]*x[1]*x[1]*exp(t)*(2*x[0] - 1)*(x[0] - 1))/12800 - (x[0]*exp(t)*(2*x[0] - 1)*(x[0] - 1)*(x[1] - 1)*(x[1] - 1))/12800 - (x[0]*x[1]*exp(t)*(2*x[0] - 1)*(2*x[1] - 2)*(x[0] - 1))/6400 - (x[1]*x[1]*exp(t)*(x[0] - 1)*(x[1] - 1)*(x[1] - 1))/6400 - (x[0]*x[0]*x[1]*x[1]*x[1]*exp(2*t)*(2*x[1] - 1)*(x[0] - 1)*(x[0] - 1)*(x[1] - 1)*(x[1] - 1)*(x[1] - 1)*(6*x[0]*x[0] - 6*x[0] + 1))/65536 + (x[0]*x[0]*x[1]*x[1]*x[1]*exp(2*t)*(2*x[0] - 1)*(x[0] - 1)*(x[1] - 1)*(x[1] - 1)*(2*x[0]*x[0] - 3*x[0] + 1)*(2*x[1]*x[1] - 3*x[1] + 1))/32768"
solution["ux"] = ux
solution["uy"] = uy
solution["p"] = p
solution["fx"] = fx
solution["fy"] = fy
solutions.append(solution)

# 1.2
solution = {}
ux = "exp(t) * sin(pi * x[0])*sin(pi * x[0]) * sin(2 * pi * x[1])"
uy = "-exp(t) * sin(2 * pi * x[0]) * sin(pi * x[1])*sin(pi * x[1])"
p = "exp(t) * (sin(pi * x[1]) - 2 / pi)"
fx = "(exp(t)*(50*sin(pi*x[0])*sin(pi*x[0])*sin(2*pi*x[1]) - pi*pi*cos(pi*x[0])*cos(pi*x[0])*sin(2*pi*x[1]) + 3*pi*pi*sin(pi*x[0])*sin(pi*x[0])*sin(2*pi*x[1]) + 100*pi*exp(t)*cos(pi*x[0])*sin(pi*x[0])*sin(pi*x[0])*sin(pi*x[0])*sin(2*pi*x[1])*sin(2*pi*x[1]) - 100*pi*exp(t)*cos(2*pi*x[1])*sin(pi*x[0])*sin(pi*x[0])*sin(2*pi*x[0])*sin(pi*x[1])*sin(pi*x[1])))/50"
fy = "(exp(t)*(50*pi*cos(pi*x[1]) - 50*sin(2*pi*x[0])*sin(pi*x[1])*sin(pi*x[1]) + pi*pi*cos(pi*x[1])*cos(pi*x[1])*sin(2*pi*x[0]) - 3*pi*pi*sin(2*pi*x[0])*sin(pi*x[1])*sin(pi*x[1]) + 100*pi*exp(t)*cos(pi*x[1])*sin(2*pi*x[0])*sin(2*pi*x[0])*sin(pi*x[1])*sin(pi*x[1])*sin(pi*x[1]) - 100*pi*exp(t)*cos(2*pi*x[0])*sin(pi*x[0])*sin(pi*x[0])*sin(pi*x[1])*sin(pi*x[1])*sin(2*pi*x[1])))/50"
solution["ux"] = ux
solution["uy"] = uy
solution["p"] = p
solution["fx"] = fx
solution["fy"] = fy
solutions.append(solution)

# 1.3
solution = {}
ux = "20 * x[0]*x[0] * (x[0] - 1)*(x[0] - 1) * x[1] * (x[1] - 1) * (2 * x[1] - 1) * t"
uy = "-20 * x[0] * (x[0] - 1) * (2 * x[0] - 1) * x[1]*x[1] * (x[1] - 1)*(x[1] - 1) * t"
p = "10 * (2 * x[0] - 1) * (2 * x[1] - 1)"
fx = "40*x[1] - (2*t*x[0]*x[0]*(2*x[1] - 1)*(x[0] - 1)*(x[0] - 1))/5 - (4*t*x[0]*x[0]*x[1]*(x[0] - 1)*(x[0] - 1))/5 - (4*t*x[0]*x[0]*(x[0] - 1)*(x[0] - 1)*(x[1] - 1))/5 - (2*t*x[0]*x[0]*x[1]*(2*x[1] - 1)*(x[1] - 1))/5 - (2*t*x[1]*(2*x[1] - 1)*(x[0] - 1)*(x[0] - 1)*(x[1] - 1))/5 + 20*x[0]*x[0]*x[1]*(2*x[1] - 1)*(x[0] - 1)*(x[0] - 1)*(x[1] - 1) - (4*t*x[0]*x[1]*(2*x[0] - 2)*(2*x[1] - 1)*(x[1] - 1))/5 - 400*t*t*x[0]*x[0]*x[0]*x[1]*x[1]*(2*x[0] - 1)*(x[0] - 1)*(x[0] - 1)*(x[0] - 1)*(x[1] - 1)*(x[1] - 1)*(6*x[1]*x[1] - 6*x[1] + 1) + 800*t*t*x[0]*x[0]*x[0]*x[1]*x[1]*(2*x[1] - 1)*(x[0] - 1)*(x[0] - 1)*(x[1] - 1)*(2*x[0]*x[0] - 3*x[0] + 1)*(2*x[1]*x[1] - 3*x[1] + 1) - 20"
fy = "40*x[0] + (2*t*x[1]*x[1]*(2*x[0] - 1)*(x[1] - 1)*(x[1] - 1))/5 + (4*t*x[0]*x[1]*x[1]*(x[1] - 1)*(x[1] - 1))/5 + (4*t*x[1]*x[1]*(x[0] - 1)*(x[1] - 1)*(x[1] - 1))/5 + (2*t*x[0]*x[1]*x[1]*(2*x[0] - 1)*(x[0] - 1))/5 + (2*t*x[0]*(2*x[0] - 1)*(x[0] - 1)*(x[1] - 1)*(x[1] - 1))/5 - 20*x[0]*x[1]*x[1]*(2*x[0] - 1)*(x[0] - 1)*(x[1] - 1)*(x[1] - 1) + (4*t*x[0]*x[1]*(2*x[0] - 1)*(2*x[1] - 2)*(x[0] - 1))/5 - 400*t*t*x[0]*x[0]*x[1]*x[1]*x[1]*(2*x[1] - 1)*(x[0] - 1)*(x[0] - 1)*(x[1] - 1)*(x[1] - 1)*(x[1] - 1)*(6*x[0]*x[0] - 6*x[0] + 1) + 800*t*t*x[0]*x[0]*x[1]*x[1]*x[1]*(2*x[0] - 1)*(x[0] - 1)*(x[1] - 1)*(x[1] - 1)*(2*x[0]*x[0] - 3*x[0] + 1)*(2*x[1]*x[1] - 3*x[1] + 1) - 20"
solution["ux"] = ux
solution["uy"] = uy
solution["p"] = p
solution["fx"] = fx
solution["fy"] = fy
solutions.append(solution)

# 1.4
# 压强空间收敛阶不稳定
solution = {}
ux = "2 * pi * sin(pi * x[0])*sin(pi * x[0]) * sin(pi * x[1]) * cos(pi * x[1]) * cos(t)"
uy = "-2 * pi * sin(pi * x[0]) * cos(pi * x[0]) * sin(pi * x[1])*sin(pi * x[1]) * cos(t)"
p = "cos(pi * x[0]) * cos(pi * x[1])"
fx = "(3*pi*pi*pi*cos(pi*x[1])*sin(pi*x[0])*sin(pi*x[0])*sin(pi*x[1])*cos(t))/25 - pi*cos(pi*x[1])*sin(pi*x[0]) - 2*pi*cos(pi*x[1])*sin(pi*x[0])*sin(pi*x[0])*sin(pi*x[1])*sin(t) + 4*pi*pi*pi*cos(pi*x[0])*sin(pi*x[0])*sin(pi*x[0])*sin(pi*x[0])*sin(pi*x[1])*sin(pi*x[1])*sin(pi*x[1])*sin(pi*x[1])*cos(t)*cos(t) - (pi*pi*pi*cos(pi*x[0])*cos(pi*x[0])*cos(pi*x[1])*sin(pi*x[1])*cos(t))/25 + 4*pi*pi*pi*cos(pi*x[0])*cos(pi*x[1])*cos(pi*x[1])*sin(pi*x[0])*sin(pi*x[0])*sin(pi*x[0])*sin(pi*x[1])*sin(pi*x[1])*cos(t)*cos(t)"
fy = "2*pi*cos(pi*x[0])*sin(pi*x[0])*sin(pi*x[1])*sin(pi*x[1])*sin(t) - (3*pi*pi*pi*cos(pi*x[0])*sin(pi*x[0])*sin(pi*x[1])*sin(pi*x[1])*cos(t))/25 - pi*cos(pi*x[0])*sin(pi*x[1]) + 4*pi*pi*pi*cos(pi*x[1])*sin(pi*x[0])*sin(pi*x[0])*sin(pi*x[0])*sin(pi*x[0])*sin(pi*x[1])*sin(pi*x[1])*sin(pi*x[1])*cos(t)*cos(t) + (pi*pi*pi*cos(pi*x[0])*cos(pi*x[1])*cos(pi*x[1])*sin(pi*x[0])*cos(t))/25 + 4*pi*pi*pi*cos(pi*x[0])*cos(pi*x[0])*cos(pi*x[1])*sin(pi*x[0])*sin(pi*x[0])*sin(pi*x[1])*sin(pi*x[1])*sin(pi*x[1])*cos(t)*cos(t)"
solution["ux"] = ux
solution["uy"] = uy
solution["p"] = p
solution["fx"] = fx
solution["fy"] = fy
solutions.append(solution)

# 非零Dirichlet边界条件
# 2.1 [0,2]x[0][-1,1]
solution = {}
ux = "2 * cos(pi * x[1]) * sin(pi * x[0]) * sin(t)"
uy = "-2 * sin(pi * x[1]) * cos(pi * x[0]) * sin(t)"
p = "2 * sin(pi * x[1]) * sin(pi * x[0]) * cos(t)"
fx = "2*cos(pi*x[1])*sin(pi*x[0])*cos(t) + 4*pi*cos(pi*x[0])*sin(pi*x[0]) - 4*pi*cos(pi*x[0])*sin(pi*x[0])*cos(t)*cos(t) + (pi*pi*cos(pi*x[1])*sin(pi*x[0])*sin(t))/25 + 2*pi*cos(pi*x[0])*sin(pi*x[1])*cos(t)"
fy = "4*pi*cos(pi*x[1])*sin(pi*x[1]) - 2*cos(pi*x[0])*sin(pi*x[1])*cos(t) - 4*pi*cos(pi*x[1])*sin(pi*x[1])*cos(t)*cos(t) - (pi*pi*cos(pi*x[0])*sin(pi*x[1])*sin(t))/25 + 2*pi*cos(pi*x[1])*sin(pi*x[0])*cos(t)"
solution["ux"] = ux
solution["uy"] = uy
solution["p"] = p
solution["fx"] = fx
solution["fy"] = fy
solutions.append(solution)

# 2.2 [0,1]x[0][-0.25,0]
solution = {}
ux = "(x[0]*x[0] * x[1]*x[1] + exp(-x[1])) * cos(2 * pi * t)"
uy = "(-2 * x[0] * x[1]*x[1]*x[1]/3 + 2 - pi * sin(pi * x[0])) * cos(2 * pi * t)"
p = "-(2 - pi * sin(pi * x[0])) * cos(2 * pi * x[1]) * cos(2 * pi * t)"
fx = "cos(2*pi*t)*cos(2*pi*t)*(- 2*x[1]*x[0]*x[0] + exp(-x[1]))*((2*x[0]*x[1]*x[1]*x[1])/3 + pi*sin(x[0]*pi) - 2) - (x[1]*x[1]*cos(2*pi*t))/50 - (cos(2*pi*t)*(2*x[0]*x[0] + exp(-x[1])))/100 - 2*pi*sin(2*pi*t)*(exp(-x[1]) + x[0]*x[0]*x[1]*x[1]) + 2*x[0]*x[1]*x[1]*cos(2*pi*t)*cos(2*pi*t)*(exp(-x[1]) + x[0]*x[0]*x[1]*x[1]) + pi*pi*cos(2*pi*t)*cos(pi*x[0])*cos(2*pi*x[1])"
fy = "2*pi*sin(2*pi*t)*((2*x[0]*x[1]*x[1]*x[1])/3 + pi*sin(x[0]*pi) - 2) - (pi*pi*pi*cos(2*pi*t)*sin(pi*x[0]))/100 - cos(2*pi*t)*cos(2*pi*t)*((2*x[1]*x[1]*x[1])/3 + pi*pi*cos(x[0]*pi))*(exp(-x[1]) + x[0]*x[0]*x[1]*x[1]) + (x[0]*x[1]*cos(2*pi*t))/25 - 2*pi*cos(2*pi*t)*sin(2*pi*x[1])*(pi*sin(pi*x[0]) - 2) + 2*x[0]*x[1]*x[1]*cos(2*pi*t)*cos(2*pi*t)*((2*x[0]*x[1]*x[1]*x[1])/3 + pi*sin(x[0]*pi) - 2)"
solution["ux"] = ux
solution["uy"] = uy
solution["p"] = p
solution["fx"] = fx
solution["fy"] = fy
solutions.append(solution)
