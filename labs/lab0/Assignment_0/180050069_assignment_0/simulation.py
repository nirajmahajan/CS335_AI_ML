import numpy as np
import matplotlib.pyplot as plt

def toss_coin():
	return int(np.random.uniform() < 0.75)

def get_expected(n = 10):
	ans = 0
	for i in range(n):
		temp_ans = 0
		prev = 0
		while(True):
			toss = toss_coin()
			temp_ans += 1
			if toss == prev and toss == 1:
				break
			else:
				prev = toss
		ans += temp_ans
	return ans / n



n_vals = [10, 100, 1000, 10000]

exp_vals = []
for ni in range(1,100):
	n = ni*100
# for n in n_vals:
	ans = get_expected(n)
	# print(ans)
	exp_vals.append(ans)
fig, ax = plt.subplots()
ax.set_title('Expected Number of Tosses vs Number of Steps')
ax.set_xlabel('Number of steps')
ax.set_ylabel('Expected Tosses')
# ax.plot(np.array(n_vals), np.array(exp_vals))
ax.plot(np.arange(99)*100, np.array(exp_vals))
plt.show()

# plot error bars
# reference https://jakevdp.github.io/PythonDataScienceHandbook/04.03-errorbars.html
vals = []
for count, n in enumerate(n_vals):
	vals.append([])
	for i in range(10):
		ans = get_expected(n)
		vals[count].append(ans)

	vals[count] = np.array(vals[count])

avgs = [x.mean() for x in vals]

plt.figure()
plt.title('Error Bars')
plt.xlabel('log scale of n_steps')
plt.ylabel('Expected tosses')
plt.xscale('log')
plt.errorbar(np.array(n_vals), np.array(avgs), yerr = np.array(avgs)-3.11111111, fmt='o', color='black',
             ecolor='lightgray', elinewidth=3, capsize=0)
plt.show()