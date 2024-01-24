import matplotlib.pyplot as plt
from onnxrt_backend_dev.monitoring.profiling import profile

def subf(x):
    return sum(x)

def fctm():
    x1 = subf([1, 2, 3])
    x2 = subf([1, 2, 3, 4])
    return x1 + x2

pr, df = profile(lambda: [fctm() for i in range(0, 1000)], as_df=True)
ax = df[['namefct', 'cum_tall']].head(n=15).set_index(
    'namefct').plot(kind='bar', figsize=(8, 3), rot=30)
ax.set_title("example of a graph")
for la in ax.get_xticklabels():
    la.set_horizontalalignment('right');
plt.show()