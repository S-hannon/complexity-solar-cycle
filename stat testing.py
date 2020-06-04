df = read_model_output()

import random
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt

for craft in ['sta', 'stb']:
    dfc = df[df.craft == craft]
    print(craft)
    for year in range(np.min(df.time).year, np.max(df.time).year + 1):
        ydf = dfc[dfc.time > datetime(year, 1, 1)]
        ydf = ydf[ydf.time < datetime(year + 1, 1, 1)]
        if len(ydf) > 0:
            stat, pval = sps.normaltest(ydf.complexity)
            stat1, pval1 = sps.skewtest(ydf.complexity)
            stat2, pval2 = sps.kurtosistest(ydf.complexity)
            print(year, pval, pval1, pval2)


sps.ks_2samp(df[df.craft == 'sta'].complexity,
             df[df.craft == 'stb'].complexity)

df = de.add_helcats_to_df(df, 'SSE Phi [deg]', 'higeocat')
df = de.add_matches_to_df(df)
df = df.dropna(subset=['match'])
a_vals, b_vals = de.get_match_vals(df, 'complexity')
sps.ttest_rel(a_vals, b_vals)


df = de.add_width_to_df(df)
for craft in ['sta', 'stb']:
    dfc = df[df.craft == craft]
    print(craft)
    corr, pval = sps.spearmanr(dfc['complexity'].values, dfc['width'].values,
                               nan_policy='omit')
    print(corr, pval)





# try shuffling indexes
colors = ['pink', 'lightskyblue']
colors2 = ['crimson', 'navy']
names = ['STEREO-A', 'STEREO-B']
plt.figure(figsize=(10, 7))
for n, craft in enumerate(['sta', 'stb']):
    runs = 1000
    c = df[df.craft == craft].complexity.values
    w = df[df.craft == craft].width.values
    real_corr, real_pval = sps.spearmanr(c, w, nan_policy='omit')
    corrs = []
    pvals = []
    for i in range(1, runs):
        random.shuffle(c)
        random.shuffle(w)
        corr, pval = sps.spearmanr(c, w, nan_policy='omit')
        corrs.append(corr)
        pvals.append(pval)
    plt.xlabel("Spearman's Rank Correlation Coefficient")
    plt.xlim((-1, 1))
    plt.ylabel("Frequency Density")
    plt.hist(corrs, color=colors[n], alpha=0.5,
             label=names[n] + ' %s\nRandomised\nPairings'%(runs))
    plt.axvline(real_corr, color=colors2[n], label=names[n])
    plt.legend(loc=0)
    




# sort out the extra comparisons
df = paired_comparison_data()
df['joint'] = df['left_subject'] + df['right_subject']
for j in np.unique(df.joint):
    dfj = df[df.joint == j]
    if len(dfj) > 1:
        df.loc[dfj.index.values[0], ['left_wins']] = sum(dfj.left_wins)
        df.loc[dfj.index.values[0], ['right_wins']] = sum(dfj.right_wins)
        df.loc[dfj.index.values[0], ['total_votes']] = sum(dfj.total_votes)
        for n in range(1, len(dfj)):
            df = df.drop(dfj.index.values[n])
            
            
import matplotlib.pyplot as plt
df12 = df[df.total_votes == 12]
plt.figure(figsize=(10,5))
plt.hist(df12['left_wins'], bins=np.arange(-0.5, 13, 1), alpha=0.5, label='Left', color='purple')
plt.hist(df12['right_wins'], bins=np.arange(-0.5, 13, 1),  alpha=0.5, label='Right')
plt.xlim((5.5, 12.5))
plt.xlabel("Number of Votes")
plt.ylabel("Frequency")
plt.legend(loc=0)


# these are NOT normally distributed
left = df12.left_wins[df12.left_wins > 6]
right = df12.right_wins[df12.right_wins > 6]
plt.figure(figsize=(10,5))
plt.hist(left, bins=np.arange(6.5, 13, 1), alpha=0.5, label='left', color='purple')
plt.hist(right, bins=np.arange(6.5, 13, 1), alpha=0.5, label='right')
plt.xlim((6.5, 12.5))

sps.epps_singleton_2samp(left, right)

# for similar results
left = df12[df12.left_wins >= 5]
left = left.left_wins[left.left_wins <= 7]
right = df12[df12.right_wins >= 5]
right = right.right_wins[right.right_wins <= 7]
plt.figure(figsize=(10,5))
plt.hist(left, bins=[4.5, 5.5, 6.5, 7.5], alpha=0.5, label='left', color='purple')
plt.hist(right, bins=[4.5, 5.5, 6.5, 7.5], alpha=0.5, label='right')
plt.xlim((4.5, 7.5))
sps.epps_singleton_2samp(left, right)


df12 = df[df.total_votes == 12]
sps.ttest_rel(df12.left_wins, df12.right_wins)

sps.median_test(df12.left_wins, df12.right_wins)


# for CMEs with similar complexity
drop = []
for i in df.index:
    if np.isnan(df.left_complexity[i]) == True:
        drop.append(i)
    elif np.isnan(df.right_complexity[i]) == True:
        drop.append(i)
    elif abs(df.left_complexity[i] - df.right_complexity[i]) > 0.5:
        drop.append(i)
df_sim = df.drop(drop)
print(len(df_sim))
sps.median_test(df12_sim.left_wins, df12_sim.right_wins)

dfa = df[df.left_craft == 'sta']
dfb = df[df.left_craft == 'stb']


X = sps.binom(12, 0.5)
X.pmf(6)
len(df12) * X.cdf(5) # expect 7228 wins each side


sps.ks_2samp(df12[df12.left_craft == 'sta'].left_wins,
             df12[df12.right_craft == 'sta'].right_wins)