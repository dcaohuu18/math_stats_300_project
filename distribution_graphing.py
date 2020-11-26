import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from scipy import stats


def binomial():
    st.header("Binomial")

    bi_n = st.number_input("Number of trials:", min_value=0, value=10, key='binom')
    bi_p = st.number_input("Probability of Success:", min_value=0.0, max_value=1.0, value=0.5, key='binom')

    bi_mean, bi_var = stats.binom.stats(bi_n, bi_p)

    # only calculate those within 4 stds:
    start = int(bi_mean - 4*np.sqrt(bi_var))
    start = start if start >= 0 else 0
    end = int(bi_mean + 4*np.sqrt(bi_var))
    end = end if end <= bi_n else bi_n

    bi_Xs =  [x for x in range(start, end+1)]
    bi_Pmfs = [stats.binom.pmf(x, bi_n, bi_p) for x in bi_Xs]

    # Graph with matplotlib:
    # st.text('')
    # fig, ax = plt.subplots(figsize=(10,5))
    # ax.bar(Xs, Pmfs)
    # if n <= 20:
        # plt.xticks(Xs)
    # st.pyplot(fig)


    # Graph with Altair
    st.text('')
    bi_prob_df = pd.DataFrame({'X': bi_Xs, 'Pmf': bi_Pmfs})
    bi_prob_hist = alt.Chart(bi_prob_df).mark_bar().encode(
                    x=alt.X('X', title="Number of successes"),
                    y=alt.Y('Pmf', title="Pmf"),
                    tooltip=['X', alt.Tooltip('Pmf', format='.4f')]
                    ).configure_axisY(titleAngle=0, titleX=-50
                    ).properties(width=600, height=400,title='Probability histogram of Binomial distribution'
                    ).interactive()
    st.altair_chart(bi_prob_hist)


def geometric():
    st.header("Geometric")

    geo_p = st.number_input("Probability of Success:", min_value=0.0, max_value=1.0, value=0.5, key='geom')
    geo_max_x = int(5/geo_p) # decide the max number of X to plot based on p

    geo_mean, geo_var = stats.geom.stats(geo_p)

    geo_Xs =  [x for x in range(1, geo_max_x+1)]
    geo_Pmfs = [stats.geom.pmf(x, geo_p) for x in geo_Xs]

    # Graph with Altair
    st.text('')
    geo_prob_df = pd.DataFrame({'X': geo_Xs, 'Pmf': geo_Pmfs})
    geo_prob_hist = alt.Chart(geo_prob_df).mark_bar().encode(
                    x=alt.X('X', title="Number of trials until 1st success"),
                    y=alt.Y('Pmf', title="Pmf"),
                    tooltip=['X', alt.Tooltip('Pmf', format='.4f')]
                    ).configure_axisY(titleAngle=0, titleX=-50
                    ).properties(width=600, height=400,title='Probability histogram of Geometric distribution'
                    ).interactive()
    st.altair_chart(geo_prob_hist)


def nbinom():
    st.header("Negative Binomial")

    nbi_p = st.number_input("Probability of Success:", min_value=0.0, max_value=1.0, value=0.5, key='nbinom')
    nbi_r = st.number_input("Required number of successes (r):", min_value=1, value=5, key='nbinom')
    ### NOTICE: in scipy.stats documentation
        # r >> n

    nbi_mean, nbi_var = stats.nbinom.stats(nbi_r, nbi_p)

    # only calculate those within 4 stds:
    start = int(nbi_r+nbi_mean - 4*np.sqrt(nbi_var))
    start = start if start >= nbi_r else nbi_r
    end = int(nbi_r+nbi_mean + 4*np.sqrt(nbi_var))

    nbi_Xs =  [x for x in range(start, end+1)]
    nbi_Pmfs = [stats.nbinom.pmf(x-nbi_r, nbi_r, nbi_p) for x in nbi_Xs]

    # Graph with Altair
    st.text('')
    nbi_prob_df = pd.DataFrame({'X': nbi_Xs, 'Pmf': nbi_Pmfs})
    nbi_prob_hist = alt.Chart(nbi_prob_df).mark_bar().encode(
                    x=alt.X('X', title="Number of trials until r-th success"),
                    y=alt.Y('Pmf', title="Pmf"),
                    tooltip=['X', alt.Tooltip('Pmf', format='.4f')]
                    ).configure_axisY(titleAngle=0, titleX=-50
                    ).properties(width=600, height=400,title='Probability histogram of Negative Binomial distribution'
                    ).interactive()
    st.altair_chart(nbi_prob_hist)


def poisson():
    st.header("Poisson")

    pois_lambda = st.number_input("Lambda λ:", min_value=0, value=5, key='pois')
    # pois_n = pois_lambda + int(10*np.ceil(pois_lambda/200)*np.log(pois_lambda))

    pois_mean, pois_var = stats.poisson.stats(pois_lambda)

    # only calculate those within 4 stds:
    start = int(pois_mean - 4*np.sqrt(pois_var))
    start = start if start >= 0 else 0
    end = int(pois_mean + 4*np.sqrt(pois_var))

    pois_Xs =  [x for x in range(start, end+1)]
    pois_Pmfs = [stats.poisson.pmf(x, pois_lambda) for x in pois_Xs]

    # Graph with Altair
    st.text('')
    pois_prob_df = pd.DataFrame({'X': pois_Xs, 'Pmf': pois_Pmfs})
    pois_prob_hist = alt.Chart(pois_prob_df).mark_bar().encode(
                        x=alt.X('X', title="Number of occurences"),
                        y=alt.Y('Pmf', title="Pmf"),
                        tooltip=['X', alt.Tooltip('Pmf', format='.4f')]
                        ).configure_axisY(titleAngle=0, titleX=-50
                        ).properties(width=600, height=400,title='Probability histogram of Poisson distribution'
                        ).interactive()
    st.altair_chart(pois_prob_hist)


def hypergeom():
    st.header("Hypergeometric")

    N = st.number_input("Total number of items (N):", min_value=0, value=20, key='hypergeom')
    K = st.number_input("Total number of successes (K):", min_value=0, max_value=N, value=int(N/2), key='hypergeom')
    n = st.number_input("Number of chosen items (n):", min_value=0, max_value=N, value=int(N/3), key='hypergeom')

    ### NOTICE: in scipy.stats documentation
        # N >> M
        # K >> n
        # n >> N

    hyper_mean, hyper_var = stats.hypergeom.stats(N, K, n)
    
    max_k = min(K, n)
    min_k = max(0, n-(N-K))

    # only calculate those within 4 stds:
    start = int(hyper_mean - 4*np.sqrt(hyper_var))
    start = start if start >= min_k else min_k
    end = int(hyper_mean + 4*np.sqrt(hyper_var))
    end = end if end <= max_k else max_k

    hyper_ks = [k for k in range(start, end+1)]
    hyper_pmfs = [stats.hypergeom.pmf(k, N, K, n) for k in hyper_ks]

    # Graph with Altair
    st.text('')
    hyper_prob_df = pd.DataFrame({'k': hyper_ks, 'pmf': hyper_pmfs})
    hyper_prob_hist = alt.Chart(hyper_prob_df).mark_bar().encode(
                        x=alt.X('k', title="Number of chosen successes"),
                        y=alt.Y('pmf', title="Pmf"),
                        tooltip=['k', alt.Tooltip('pmf', format='.4f')]
                        ).configure_axisY(titleAngle=0, titleX=-50
                        ).properties(width=600, height=400,title='Probability histogram of Hypergeometric distribution'
                        ).interactive()
    st.altair_chart(hyper_prob_hist)


def uniform():
    st.header("Uniform")

    uni_b = st.number_input("Upper bound (b):", value=10.0, key='uniform')
    uni_a = st.number_input("Lower bound (a):", max_value=uni_b, value=uni_b//2, key='uniform')

    st.text('')
    mark_area = st.checkbox("Mark area", key='uniform')

    # Graph with Altair
    st.text('')
    uni_pdf_df = pd.DataFrame({'x': np.linspace(uni_a, uni_b, 101), 'pdf': 101*[1/(uni_b-uni_a)]})
    uni_pdf_hist = alt.Chart(uni_pdf_df).mark_line().encode(
                        x=alt.X('x', title="X", scale=alt.Scale(domain=[uni_a-0.1, uni_b+0.1])),
                        y=alt.Y('pdf', title="Pdf", scale=alt.Scale(domain=[0, 1.5/(uni_b-uni_a)])))

    if mark_area:
        uni_pdf_hist = uni_pdf_hist.encode(x=alt.X('x', title="X")) # set the scale domain to default
        uni_area = alt.Chart(uni_pdf_df).mark_bar(size=2).encode(x='x', y='pdf')
    else:
        uni_area = alt.LayerChart() # empty chart

    complete_graph = alt.layer(uni_pdf_hist, uni_area
                        ).configure_axisY(titleAngle=0, titleX=-50
                        ).properties(width=600, height=400,title='Pdf of Uniform distribution'
                        ).interactive()
    
    st.altair_chart(complete_graph)


def normal():
    st.header("Normal")

    norm_mu = st.number_input("Mean (μ):", value=0.0, key='norm')
    norm_sigma = st.number_input("Standard deviation (σ):", min_value=0.0, value=1.0, key='norm')

    # only calculate those within 4 stds:
    start = norm_mu - 4*norm_sigma
    end = norm_mu + 4*norm_sigma

    norm_Xs = np.linspace(start, end, 101) # take 101 equally spaced xs to smoothen the curve 
    norm_Pdfs = [stats.norm.pdf(x, norm_mu, norm_sigma) for x in norm_Xs]

    st.text('')
    mark_area = st.checkbox("Mark area", key='norm')

    # Graph with Altair
    st.text('')
    norm_pdf_df = pd.DataFrame({'x': norm_Xs, 'pdf': norm_Pdfs})
    norm_pdf_hist = alt.Chart(norm_pdf_df).mark_line().encode(
                        x=alt.X('x', title="X",scale=alt.Scale(domain=[norm_Xs[0]-0.1, norm_Xs[-1]+0.1])),
                        y=alt.Y('pdf', title="Pdf"))

    if mark_area:
        norm_area = alt.Chart(norm_pdf_df).mark_bar(size=2).encode(x='x', y='pdf')
    else:
        norm_area = alt.LayerChart() # empty chart
    
    complete_graph = alt.layer(norm_pdf_hist, norm_area
                        ).configure_axisY(titleAngle=0, titleX=-50
                        ).properties(width=600, height=400,title='Pdf of Normal distribution'
                        ).interactive()
    
    st.altair_chart(complete_graph)


DIST_FUNC_MAP = {
    "Binomial": binomial,
    "Geometric": geometric,
    "Negative Binomial": nbinom,
    "Poisson": poisson,
    "Hypergeometric": hypergeom,
    "Uniform": uniform,
    "Normal": normal
}

def main():
    # Title 
    st.title('Distribution Graphing Calculator')

    # About:
    st.sidebar.header('About')
    st.sidebar.info('Graphing tool for some fundamental statistical distributions.')

    # Select type:
    dist_types = ['Discrete', 'Continuous']
    dist_type_choice = st.sidebar.radio('Type', dist_types)

    # Select distribution:
    discr_dist_list = ['Binomial', 'Geometric', 'Negative Binomial', 'Poisson', 'Hypergeometric']
    cont_dist_list = ['Uniform', 'Normal']

    if dist_type_choice == 'Discrete': 
    	dist_choice = st.selectbox('Select your distribution:', discr_dist_list)
    else:
    	dist_choice = st.selectbox('Select your distribution:', cont_dist_list)

    DIST_FUNC_MAP[dist_choice]()


if __name__ == '__main__':
    main()

