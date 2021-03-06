import numpy as np
from scipy import stats
from collections import namedtuple

# Source: https://stackoverflow.com/questions/37676539/numpy-padding-matrix-of-different-row-size
#@profile
def pad_with_nan(M):
    """Appends the minimal required amount of zeroes at the end of each 
     array in the jagged array `M`, such that `M` looses its jagedness."""

    maxlen = max(len(r) for r in M)

    n = [np.pad(row, (0, maxlen - len(row)))
	    for row in M]

    np.asarray(n)

    Z = np.zeros((len(M), maxlen))
    for enu, row in enumerate(M):
        Z[enu, :len(row)] += row 
    return Z

#@profile
def alexander_govern_orig(a): 
	a = np.asarray(a)


	#Based on ttest_1samp
	#Works as expected in multiple cases.
	def calc_t(a): 
		n = a.shape[1]

		df = n - 1
		X = np.nanmean(a, 1)

		s = np.nanvar(a, 1, ddof=1)
		S = np.sqrt(s / n)

		w = 1/S**2 / np.sum(1/S**2)	#Weights

		u = np.sum(w * X)
		
		t = (X - u) / S
		print(X, u, S)
		return t, df

	#Below follows nomenclature in the evaluating AG paper
	def calc_zi(ti, vi):
		a = vi - .5
		b = 48*a**2
		c=(a*np.log(1 + ti**2/vi))**.5

		#Calc Z. tX is Term in equation
		t0 = c
		t1 = (c**3 + 3*c) / b
		t3 = (4*c**7 + 33*c**5 + 240*c**3 + 855*c) / (10*b**2 + 8*b*c**4 + 1000*b)

		return t0 + t1 - t3



	t, v = calc_t(a) # T is arr of T stats, v is dimension - 1
	print(t,v)

	z = calc_zi(t, v)
	print(z)

	A = np.sum(z**2)
	print(A)

	return A
	
#@profile
def alexander_govern_nan_fill(a): 
	a = pad_with_nan(a)
	a = np.asarray(a)


	#Based on ttest_1samp
	#Works as expected in multiple cases.
	#@profile
	def calc_t(a): 
		#https://stackoverflow.com/questions/44525825/count-number-of-non-nan-values-in-array
		n = (~np.isnan(a)).sum(1)
		#print(n)

		df = n - 1
		X = np.nanmean(a, 1)

		s = np.nanvar(a, 1, ddof=1)
		S = np.sqrt(s / n)

		w = 1/S**2 / np.sum(1/S**2)	#Weights

		u = np.sum(w * X)
		
		t = (X - u) / S
		#print(X, u, S)
		return t, df

	#Below follows nomenclature in the evaluating AG paper
	def calc_zi(ti, vi):
		a = vi - .5
		b = 48*a**2
		c=(a*np.log(1 + ti**2/vi))**.5

		#Calc Z. tX is Term in equation
		t0 = c
		t1 = (c**3 + 3*c) / b
		t3 = (4*c**7 + 33*c**5 + 240*c**3 + 855*c) / (10*b**2 + 8*b*c**4 + 1000*b)

		return t0 + t1 - t3



	t, v = calc_t(a) # T is arr of T stats, v is dimension - 1
	#print(t,v)

	z = calc_zi(t, v)
	#print(z)

	A = np.sum(z**2)
	#print(A)

	return A



"""

for a_slice in a:
	
"""
AlexanderGovernResult = namedtuple("AlexanderGovernResult", ("statistic",
                                                             "pvalue"))

@profile
def alexander_govern_progressive(*args):
	
	#Custom SEM function using cached vars is ~4x faster
	def standard_error(data, mean, length):
		sum_ = np.sum((data - mean)**2)
		w = length * (length - 1)
		return (sum_ / w) ** .5


	def calc_z2(a_slice, var_w, S, mean, length):
		df = length - 1
		t = (mean - var_w) / S

		a_ = df - .5
		b = 48*a_**2
		c=(a_*np.log(1 + t**2/df))**.5

		t0 = c
		t1 = (c**3 + 3*c) / b
		t3 = (4*c**7 + 33*c**5 + 240*c**3 + 855*c) / (10*b**2 + 8*b*c**4 + 1000*b)

		return (t0 + t1 - t3)**2

	#Using fromiter vs asarray is slightly faster
	#a_test = [np.asarray(A) for A in args]
	a = [np.fromiter(A, float) for A in args]


	lens = np.asarray([len(A) for A in a])
	#lens = np.vectorize(lambda A: len(A))(a)

	means = np.asarray([np.mean(A) for A in a])
	#means = np.vectorize(lambda A: np.mean(a))(a)

	#stats.sem is slow.
	S = np.vectorize(standard_error)(a, means, lens)
	#S = np.vectorize(lambda A: stats.sem(A))(a)

	w = 1/S**2 / np.sum(1/S**2)

	var_w = np.sum(w * means)

	# For small datasets, list comprehension is faster. 
	# For large sets, vectorization is
	Z = np.vectorize(calc_z2)(a, var_w, S, means, lens)
	#Z = [calc_z2(a, var_w, S, means, lens) for a, S, means, lens in zip(a, S, means, lens)]


	Z = np.sum(Z)

	p = stats.distributions.chi2.sf(Z, len(a) - 1)

	return AlexanderGovernResult(Z, p)




@profile
def AlexanderGovern(*args):
    if len(args) < 2:
        raise TypeError(f"2 or more inputs required, got {len(args)}")

    args = [np.asarray(arg) for arg in args]

    # The following formula numbers reference the equation described on
    # page 92 by Alexander, Govern. Formulas 5, 6, and 7 describe other
    # tests that serve as the basis for equation (8) but are not needed
    # to perform the test.

    # (1) determine standard errors for each sample
    def standard_error(data):
        # this is much faster than using stats.sem for some reason
        mean = np.mean(data)
        sum_ = np.sum((data - mean)**2)
        n = len(data)
        w = n * (n - 1)
        return (sum_ / w) ** .5
    standard_errors = [standard_error(arg) for arg in args]
    

    # precalculate weighted sum for following step
    weight_denom = np.sum(1 / np.square(standard_errors))

    # (2) define a weight for each samlple
    weights = [(1 / s**2) / weight_denom for s in standard_errors]

    # precalculate means of each sample
    means = np.asarray([np.mean(arg) for arg in args])
    # (3) determine variance-weighted estimate of the common mean
    var_w = np.sum(weights * means)

    # (4) determine one-sample t statistic for each group
    t_stats = [((mean - var_w)/s) for mean, s in zip(means, standard_errors)]
    

    # calculate parameters to be used in transformation
    v = [len(k) - 1 for k in args]
    a = [v_i - .5 for v_i in v]
    b = [48 * a_i**2 for a_i in a]
    c = [((a_i * np.log(1 + (t_i ** 2)/v_i))**.5)
         for a_i, t_i, v_i in zip(a, t_stats, v)]

    # (8) perform a normalizing transformation on t statistic
    z = [(c_i + ((c_i**3 + 3*c_i)/b_i) -
          ((4*c_i**7 + 33*c_i**5 + 240*c_i**3 + 855*c_i) /
           (10*b_i**2 + 8*b_i*c_i**4 + 1000*b_i)))
         for c_i, b_i in zip(c, b)]

    # (9) calculate statistic
    A = np.sum(np.square(z))

    # "[the p value is determined from] central chi-square random deviates
    # with n_i - 1 degrees of freedom". Alexander, Govern (94)
    p = stats.distributions.chi2.sf(A, len(args) - 1)
    return AlexanderGovernResult(A, p)





y = [482.43, 484.36, 488.84, 495.15, 495.24, 502.69, 504.62, 518.29, 519.10, 
524.10, 524.12, 531.18, 548.42, 572.10, 584.68, 609.09, 609.53, 666.63, 676.40]

m = [335.59, 338.43, 353.54, 404.27, 437.5, 469.01, 485.85, 487.3, 493.08, 
494.31, 499.1, 886.41]

o = [519.01, 528.5, 530.23, 536.03, 538.56, 538.83, 557.24, 558.61, 558.95, 
565.43, 586.39, 594.69, 629.22, 645.69, 691.84]

a = [y,m,o]

valMult = 500

sampMin = 100
sampMax = 1000

colMin = 10000
colMax = 100000

np.random.seed(1235)
r = [(np.random.rand(np.random.randint(colMin, colMax)) * valMult).tolist() 
		for i in range(np.random.randint(sampMin, sampMax))]

@profile
def run() :
	
	for i in range(10):
		print(alexander_govern_progressive(*r))
		print(AlexanderGovern(*r))
	


run()
#print(pad_with_nan(a))