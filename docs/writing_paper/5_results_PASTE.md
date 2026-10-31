# Mục 5 Results — nội dung để paste vào Overleaf

Mục 5 là **kết quả danh mục**: so sánh tổng thể với EB (5.1), độ bền trước nhiễu (5.2), theo thời gian (5.3), thảo luận và giới hạn (5.4). Cách làm, đối chứng, chỉ số, độ chính xác dự báo và ablation đã ở mục 4, nên mục 5 chỉ trỏ lại. Mô hình của mình luôn đứng **cuối** trong mọi bảng và mọi đoạn so sánh. Phân tích mới ở 5.1.3: nơi hai phương pháp bất đồng.

## Cách dùng

1. Paste khối LaTeX này **ngay sau khối của mục 4**, trước `\section{Conclusion}\label{sec6}`.
2. Hình `cumret_top1.png` đã có trong `overleaf_project/images/`.
3. Không có BibTeX mới.

## Cấu trúc của mục 5

| Mục | Tên | Nội dung | Bảng / hình |
|---|---|---|---|
| 5.1 | Overall performance comparison | | |
| 5.1.1 | Baseline comparison | Sharpe của 4 phương pháp, mô hình ở cuối | Bảng 3 |
| 5.1.2 | Selection performance | lãi, t-stat, p-value | Bảng 4 |
| 5.1.3 | Where the methods disagree | chiến lược chỉ EB chọn / cả hai / chỉ mô hình chọn | Bảng 5 |
| 5.1.4 | Summary of the comparison | điểm mạnh của mô hình so với tất cả | |
| 5.2 | Robustness to noise strategies | | |
| 5.2.1 | Ticker control | họ ticker làm đối chứng nhiễu | |
| 5.2.2 | Sensitivity analysis | tập gộp thêm ticker so với chỉ kế toán | Bảng 6 |
| 5.2.3 | Summary of the noise analysis | | |
| 5.3 | Temporal performance analysis | | |
| 5.3.1 | Cumulative returns | lãi cộng dồn | Hình 1 |
| 5.3.2 | Year-by-year performance | từng năm lịch | |
| 5.3.3 | Performance stability | 4 giai đoạn con | Bảng 7 |
| 5.4 | Discussion and limitations | | |
| 5.4.1 | Interpretation | | |
| 5.4.2 | Practical implications | | |
| 5.4.3 | Limitations | | |

---

## Bản tiếng Anh (paste vào LaTeX)

```latex
\section{Results}\label{sec5}

Can the proposed model select better strategies than EB mining using only information available in real time? We answer this question with the portfolio tests of Section \ref{subsec4_2}.

\subsection{Overall performance comparison}\label{subsec5_1}

\subsubsection{Baseline comparison}
Table \ref{tab:sharpe} compares the four methods of Section \ref{subsec4_2}, ordered from the weakest to the proposed model. The baseline model has the lowest Sharpe ratio in every cell, so the improvement of the proposed model comes from the design of Section \ref{subsec3} and not from the FinCast representations alone (for the top 1\%, 1.95 against 1.34 in the development period and 1.11 against 0.61 in the holdout period). Naive mining is the closest competitor of the proposed model: it equals the proposed model for the development top 1\% (1.95) and is higher for the development top 5\% (1.99). The proposed model has the highest Sharpe ratio in all three holdout cells and in the full-period top 1\% and 5\%, and the highest or tied-highest value in 6 of the 9 cells. The exception that favors EB mining is the development top 10\% (1.86 against 1.82).

\begin{table}[h]
\caption{Annualized Sharpe ratios of equal-weighted long-short portfolios from the top 1\%, 5\% and 10\% of equal-weighted accounting strategies. The development period is 1983--2004 and the holdout period is 2005--2020.}\label{tab:sharpe}%
\small
\begin{tabular}{@{}lccccccccc@{}}
\toprule
 & \multicolumn{3}{c}{Development} & \multicolumn{3}{c}{Holdout} & \multicolumn{3}{c}{Full} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}
Method & 1\% & 5\% & 10\% & 1\% & 5\% & 10\% & 1\% & 5\% & 10\% \\
\midrule
Baseline model & 1.34 & 1.23 & 1.16 & 0.61 & 0.51 & 0.54 & 1.07 & 0.97 & 0.94 \\
Naive mining ($|t|$ rule) & \textbf{1.95} & \textbf{1.99} & 1.85 & 0.84 & 0.75 & 0.83 & 1.52 & 1.51 & 1.47 \\
EB mining & 1.55 & 1.91 & \textbf{1.86} & 0.82 & 0.78 & 0.86 & 1.27 & 1.48 & \textbf{1.48} \\
Proposed model & \textbf{1.95} & 1.95 & 1.82 & \textbf{1.11} & \textbf{0.88} & \textbf{0.88} & \textbf{1.62} & \textbf{1.54} & 1.46 \\
\botrule
\end{tabular}
\end{table}

\subsubsection{Selection performance}
Table \ref{tab:selection} reports returns, t-stats and the significance of the Sharpe ratio differences. The top 1\% portfolio of EB mining earns 8.95\% per year in the development period and 2.79\% in the holdout period, with t-stats of 7.26 and 3.27. The proposed model earns 9.36\% and 3.54\%, with t-stats of 9.15 and 4.45. The Sharpe ratio difference from EB mining is significant in the development ($p = 0.032$), holdout ($p = 0.028$) and full ($p = 0.005$) periods, and a block bootstrap by year gives 0.059, 0.051 and 0.010. Against naive mining, the proposed model is indistinguishable in the development period ($p = 0.97$) and higher in the holdout period ($p = 0.028$). For the top 5\% and 10\%, no difference from EB mining is significant, so the advantage concentrates at the top of the ranking, which matters most to a data miner who buys few strategies.

\begin{table}[h]
\caption{Mean return (percent per year), t-stat, and $p$-values of the Sharpe ratio difference between the proposed model and each method (HAC test).}\label{tab:selection}%
\small
\begin{tabular}{@{}llcccccc@{}}
\toprule
 & & \multicolumn{2}{c}{EB mining} & \multicolumn{2}{c}{Proposed model} & \multicolumn{2}{c}{$p$-value vs.} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}
Portfolio & Period & Return & t-stat & Return & t-stat & EB & Naive \\
\midrule
Top 1\%  & Development & 8.95 & 7.26 & 9.36 & 9.15 & 0.032 & 0.970 \\
         & Holdout     & 2.79 & 3.27 & 3.54 & 4.45 & 0.028 & 0.028 \\
         & Full        & 6.36 & 7.84 & 6.91 & 9.97 & 0.005 & 0.104 \\
Top 5\%  & Development & 8.42 & 8.97 & 8.05 & 9.16 & 0.625 & 0.445 \\
         & Holdout     & 2.29 & 3.11 & 2.44 & 3.50 & 0.330 & 0.189 \\
         & Full        & 5.84 & 9.11 & 5.69 & 9.46 & 0.317 & 0.646 \\
Top 10\% & Development & 7.73 & 8.74 & 7.13 & 8.51 & 0.315 & 0.459 \\
         & Holdout     & 2.26 & 3.42 & 2.16 & 3.51 & 0.828 & 0.653 \\
         & Full        & 5.43 & 9.11 & 5.04 & 8.98 & 0.626 & 0.838 \\
\botrule
\end{tabular}
\end{table}

\subsubsection{Where the methods disagree}
The proposed model and EB mining share only part of their top 1\% portfolios: averaged over the 38 years, 51.5\% of the strategies in the top 1\% of the proposed model are also in that of EB mining. Table \ref{tab:disagree} separates the strategies selected only by EB mining, by both methods and only by the proposed model (about 122, 130 and 122 strategies per year). The strategies selected only by the proposed model earn a Sharpe ratio of 1.63 over the full period, against 0.85 for those selected only by EB mining, and the difference is significant in the development ($p < 0.001$), holdout ($p = 0.038$) and full ($p < 0.001$) periods. Their mean return is also higher (6.66\% against 5.81\% per year), although this difference alone is not statistically significant ($t = 0.99$), and their t-stat is much higher (10.02 against 5.24), so their advantage lies mainly in more stable returns. Thus, where the two methods disagree, the choices of the proposed model are the better ones. This suggests that the return history carries information about future performance that the summary statistics used by EB mining do not.

\begin{table}[h]
\caption{Top 1\% portfolios split by which method selects the strategy. Mean return is in percent per year.}\label{tab:disagree}%
\small
\begin{tabular}{@{}lccccccccc@{}}
\toprule
 & \multicolumn{3}{c}{Development} & \multicolumn{3}{c}{Holdout} & \multicolumn{3}{c}{Full} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}
Selected by & Return & t-stat & Sharpe & Return & t-stat & Sharpe & Return & t-stat & Sharpe \\
\midrule
EB mining only & 8.37 & 4.69 & 1.00 & 2.29 & 2.53 & 0.63 & 5.81 & 5.24 & 0.85 \\
Both methods & 9.77 & 8.98 & 1.91 & 3.45 & 3.80 & 0.95 & 7.11 & 9.47 & 1.54 \\
Proposed model only & 8.85 & 9.14 & \textbf{1.95} & 3.65 & 4.56 & \textbf{1.14} & 6.66 & 10.02 & \textbf{1.63} \\
\botrule
\end{tabular}
\end{table}

\subsubsection{Summary of the comparison}
The proposed model selects the top 1\% of strategies with a higher Sharpe ratio, mean return and t-stat than EB mining in the development, holdout and full periods, and the Sharpe ratio differences are statistically significant. In the holdout period, it has the highest Sharpe ratio of the four methods at every portfolio size. The strategies that it selects beyond those of EB mining earn a Sharpe ratio nearly twice as high (1.63 against 0.85). For the top 5\% and 10\%, the methods are statistically indistinguishable. The model also treats ticker strategies, which carry no predictability, as noise at the top of its ranking: they are only 0.19\% of its top 1\% portfolio, although they are 43.5\% of the pool (Section \ref{subsec5_2}).

\subsection{Robustness to noise strategies}\label{subsec5_2}

\subsubsection{Ticker control}
Ticker strategies carry no predictability \cite{chen2025highthroughputassetpricing}, so they provide a control for noise: a method that mistakes noise for signal would select them. We apply the trained model to the pool that adds the equal-weighted ticker strategies, without retraining and without the model having seen ticker strategies. They make up 43.5\% of the pool, so a random selection would hold this share. They are 0.19\%, 3.43\% and 9.30\% of the top 1\%, 5\% and 10\% of the proposed model (Table \ref{tab:ticker}). The 33 ticker strategies in the top 1\% over the 38 years earn $-0.07\%$ per month, which is consistent with noise, and the IC of the model within ticker strategies is close to zero ($-0.035$). Thus, the proposed model recognizes ticker strategies as noise at the top of its ranking.

\subsubsection{Sensitivity analysis}
Table \ref{tab:ticker} compares the methods on the pool that adds ticker strategies with the accounting family alone. Adding a noise family changes the top 1\% Sharpe ratio of the proposed model by $+0.07$ in the development period and $-0.06$ in the holdout period, and that of naive mining by $+0.06$ and $-0.09$. For EB mining, which is re-estimated on the pool, the change is $+0.17$ and $-0.07$. The proposed model has the highest top 1\% Sharpe ratio on the pool in both periods (2.02 and 1.05). Ticker strategies enter the top 5\% and 10\% of the proposed model more often than those of EB mining (Table \ref{tab:ticker}), so EB mining shrinks noise more strongly beyond the top 1\%.

\begin{table}[h]
\caption{Sensitivity to the ticker family. Share of ticker strategies in the top 1\%, 5\% and 10\% portfolios of the pool, and top 1\% Sharpe ratios on the pool and on the accounting family alone.}\label{tab:ticker}%
\small
\begin{tabular}{@{}lccccccc@{}}
\toprule
 & \multicolumn{3}{c}{Ticker share (\%)} & \multicolumn{2}{c}{Sharpe, pool} & \multicolumn{2}{c}{Sharpe, accounting only} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}
Method & 1\% & 5\% & 10\% & Dev. & Hold. & Dev. & Hold. \\
\midrule
Naive mining ($|t|$ rule) & 0.00 & 0.52 & 2.88 & 2.01 & 0.75 & 1.95 & 0.84 \\
EB mining & 0.00 & 0.18 & 1.25 & 1.72 & 0.75 & 1.55 & 0.82 \\
Proposed model & 0.19 & 3.43 & 9.30 & \textbf{2.02} & \textbf{1.05} & 1.95 & 1.11 \\
\botrule
\end{tabular}
\end{table}

\subsubsection{Summary of the noise analysis}
The proposed model almost never selects ticker strategies for its top 1\%, as EB mining, and it keeps the highest top 1\% Sharpe ratio when the noise strategies are added to the pool. Beyond the top 1\%, EB mining shrinks the noise strategies more strongly.

\subsection{Temporal performance analysis}\label{subsec5_3}

\subsubsection{Cumulative returns}
Figure \ref{fig:cumret} plots the cumulative returns of the top 1\% portfolios. The proposed model stays above EB mining for most of the sample and ends at 262\% against 241\%, and the baseline model ends at 120\%. Naive mining reaches a similar cumulative return (264\%) with larger fluctuations, so its holdout Sharpe ratio is lower (Table \ref{tab:sharpe}). All methods earn most of their returns before 2004, and the Sharpe ratio of every method is lower in the holdout period, consistent with the decline in predictability documented by \cite{chen2025highthroughputassetpricing}. That study associates the decline with the rise of information technology around 2004 and with limited attention and the slow diffusion of information.

\begin{figure}[h]
\centering
\includegraphics[width=0.9\textwidth]{images/cumret_top1.png}
\caption{Cumulative return (percent) of the top 1\% portfolios. The shaded area is the holdout period.}\label{fig:cumret}
\end{figure}

\subsubsection{Year-by-year performance}
The proposed model does not dominate year by year. Its calendar-year return exceeds that of EB mining in only 21 of the 38 years (13 of 22 in the development period and 8 of 16 in the holdout period), and both methods have a positive return in 33 years. The difference arises in the bad years. The worst calendar-year return is $-2.5\%$ for the proposed model (2005) against $-8.1\%$ for EB mining (2020), and the maximum drawdown of the cumulative return is 6.6 percentage points against 10.3. In the holdout period, the standard deviation of calendar-year returns is 3.5\% against 4.1\%, while in the development period it is slightly higher (6.0\% against 5.5\%), in line with its higher return.

\subsubsection{Performance stability}
Table \ref{tab:subperiod} splits the sample into four subperiods. The top 1\% Sharpe ratio of the proposed model is higher than that of EB mining in all four, and higher than that of naive mining in the last three. This includes the last subperiod, in which predictability is weakest (0.99 against 0.58).

\begin{table}[h]
\caption{Top 1\% Sharpe ratio by subperiod.}\label{tab:subperiod}%
\small
\begin{tabular}{@{}llcccc@{}}
\toprule
Measure & Method & 1983--1994 & 1995--2004 & 2005--2012 & 2013--2020 \\
\midrule
Sharpe ratio & Naive mining ($|t|$ rule) & 2.26 & 1.89 & 1.03 & 0.66 \\
 & EB mining & 2.00 & 1.52 & 1.11 & 0.58 \\
 & Proposed model & 2.17 & 1.97 & 1.25 & 0.99 \\
\botrule
\end{tabular}
\end{table}

\subsection{Discussion and limitations}\label{subsec5_4}

\subsubsection{Interpretation}
The advantage of the proposed model concentrates in one place: the top of the ranking, and it persists after 2004. This is consistent with its design. The label of Equation \eqref{eq:label} compares strategies within a year, and the ablation of Section \ref{subsec4_3} shows that this label and the pretrained encoder contribute to the result, while it does not distinguish ISAB, the Huber loss or the year weights from run-to-run variation. Selecting strategies with large in-sample returns also selects large sampling error, and EB mining corrects this through shrinkage of the t-stat \cite{chen2025highthroughputassetpricing}. Its forecasts are therefore closer to realized returns in magnitude, which may explain its better calibration and higher development IC. The t-stat already carries most of the information for the top of the ranking, which explains why naive mining performs well, consistent with Proposition 1 of \cite{chen2025highthroughputassetpricing}. The proposed model nonetheless adds to it. Two strategies with the same t-stat can differ in how their returns are distributed over time, and the strategies that only the proposed model selects earn a much higher Sharpe ratio than those that only EB mining selects (Section \ref{subsec5_1}), so the return history may carry additional information.

\subsubsection{Practical implications}
For a data miner who holds the top 1\% (about 257 strategies in a typical year) for 12 months, the proposed model offers a higher return per unit of risk and smaller losses in bad years than EB mining, using only information available in real time. Because it ranks better but estimates return magnitudes worse, it fits the task of choosing which strategies to hold, and not the task of forecasting how much they will earn. The holdout return of the top 1\% portfolio is 3.5\% per year before transaction costs, lower than the development return, so the return after 2004 is modest in size.

\subsubsection{Limitations}
First, the proposed model was selected on the development period after exploring about 60 configurations, so the $p$-values are descriptive and not adjusted for multiple testing. Second, retraining the final configuration gives a lower holdout Sharpe ratio on average (0.84 over three runs, against 0.96 for the stored model), so the stored model may be somewhat better than a typical retraining. Third, our evidence concerns the equal-weighted accounting family, which accounts for most of the top 1\% of EB mining, and the advantage over EB mining is statistically supported only for the top 1\% portfolio; it does not extend to other markets. Fourth, FinCast is pretrained on external data whose periods may overlap with our forecasting years, and we cannot rule out that this gives the model some information about them. Finally, the portfolios are before transaction costs.
```

---

## Bản tiếng Việt (dịch sát bản tiếng Anh, để bạn đọc, không paste)

### 5 Results (Kết quả)

Mô hình đề xuất có chọn được chiến lược tốt hơn EB chỉ bằng thông tin có sẵn theo thời gian thực không? Ta trả lời bằng các phép thử danh mục ở mục 4.2.

#### 5.1 Overall performance comparison (So sánh hiệu quả tổng thể)

**5.1.1 Baseline comparison (So với các phương pháp đối chứng).** Bảng 3 so sánh bốn phương pháp ở mục 4.2, xếp từ yếu nhất đến mô hình đề xuất. Mô hình baseline có Sharpe thấp nhất ở mọi ô, nên phần cải thiện của mô hình đề xuất đến từ thiết kế ở mục 3.2 chứ không chỉ từ biểu diễn FinCast (top 1%: 1,95 so với 1,34 ở dev và 1,11 so với 0,61 ở holdout). Naive mining là đối thủ gần nhất của mô hình đề xuất: bằng mô hình ở dev top 1% (1,95) và cao hơn ở dev top 5% (1,99). Mô hình đề xuất có Sharpe cao nhất ở cả ba ô holdout và ở top 1%, 5% của cả kỳ, và cao nhất hoặc ngang cao nhất ở 6/9 ô. Ngoại lệ nghiêng về EB là dev top 10% (1,86 so với 1,82).

**5.1.2 Selection performance (Hiệu quả chọn chiến lược).** Bảng 4 báo lãi, t-stat và ý nghĩa của chênh lệch Sharpe. Danh mục top 1% của EB lãi 8,95%/năm ở dev và 2,79% ở holdout, t-stat 7,26 và 3,27. Mô hình đề xuất lãi 9,36% và 3,54%, t-stat 9,15 và 4,45. Chênh lệch Sharpe so với EB có ý nghĩa ở dev ($p=0,032$), holdout (0,028) và cả kỳ (0,005); bootstrap theo khối năm cho 0,059, 0,051, 0,010. So với naive mining, mô hình không khác biệt ở dev ($p=0,97$) và cao hơn ở holdout ($p=0,028$). Ở top 5% và 10% không có chênh lệch nào so với EB có ý nghĩa, nên lợi thế tập trung ở đầu bảng xếp hạng, là phần quan trọng nhất với người khai thác dữ liệu vì họ mua ít chiến lược.

**5.1.3 Where the methods disagree (Nơi hai phương pháp bất đồng).** Mô hình đề xuất và EB chỉ chia sẻ một phần danh mục top 1%: trung bình 38 năm, 51,5% chiến lược trong top 1% của mô hình cũng nằm trong top 1% của EB. Bảng 5 tách các chiến lược chỉ EB chọn, cả hai cùng chọn, và chỉ mô hình đề xuất chọn (khoảng 122, 130 và 122 chiến lược mỗi năm). Các chiến lược chỉ mô hình chọn đạt Sharpe 1,63 cả kỳ, so với 0,85 của các chiến lược chỉ EB chọn, và chênh lệch có ý nghĩa ở dev ($p<0,001$), holdout ($p=0,038$) và cả kỳ ($p<0,001$). Lãi trung bình của chúng cũng cao hơn (6,66% so với 5,81%/năm), dù riêng chênh lệch này chưa có ý nghĩa thống kê ($t=0,99$), và t-stat cao hơn nhiều (10,02 so với 5,24), nên lợi thế chủ yếu nằm ở lãi ổn định hơn. Vì vậy ở chỗ hai phương pháp bất đồng, lựa chọn của mô hình đề xuất là lựa chọn tốt hơn. Điều này gợi ý lịch sử lợi suất chứa thông tin về hiệu quả tương lai mà các thống kê tóm tắt EB dùng không có.

**5.1.4 Summary of the comparison (Tóm tắt so sánh).** Mô hình đề xuất chọn top 1% với Sharpe, lãi trung bình và t-stat cao hơn EB ở dev, holdout và cả kỳ, và chênh lệch Sharpe có ý nghĩa thống kê. Ở holdout, mô hình đề xuất có Sharpe cao nhất trong bốn phương pháp ở mọi cỡ danh mục. Các chiến lược mà mô hình chọn thêm so với EB đạt Sharpe cao gần gấp đôi (1,63 so với 0,85). Ở top 5% và 10%, các phương pháp không phân biệt được về thống kê. Mô hình cũng coi ticker, vốn không có khả năng dự báo, là nhiễu ở đầu bảng xếp hạng: chỉ 0,19% top 1% là ticker, dù ticker chiếm 43,5% tập gộp (mục 5.2).

#### 5.2 Robustness to noise strategies (Độ bền trước chiến lược nhiễu)

**5.2.1 Ticker control (Đối chứng ticker).** Chiến lược ticker không có khả năng dự báo, nên là đối chứng cho nhiễu: một phương pháp nhầm nhiễu thành tín hiệu sẽ chọn chúng. Ta áp mô hình đã học vào tập gộp thêm các chiến lược ticker chia đều, không học lại và mô hình chưa từng thấy ticker. Ticker chiếm 43,5% tập gộp, nên chọn bừa sẽ giữ đúng tỉ lệ này. Ticker chiếm 0,19%, 3,43%, 9,30% của top 1%, 5%, 10% của mô hình (Bảng 6). 33 chiến lược ticker lọt vào top 1% trong 38 năm có lãi −0,07%/tháng, phù hợp với nhiễu, và IC của mô hình trong riêng ticker xấp xỉ 0 (−0,035). Vì vậy mô hình đề xuất nhận ra ticker là nhiễu ở đầu bảng xếp hạng.

**5.2.2 Sensitivity analysis (Phân tích độ nhạy).** Bảng 6 so sánh các phương pháp trên tập gộp thêm ticker với họ kế toán riêng. Thêm một họ nhiễu làm Sharpe top 1% của mô hình đổi +0,07 ở dev và −0,06 ở holdout, của naive mining +0,06 và −0,09. Với EB, được ước lượng lại trên tập gộp, thay đổi là +0,17 và −0,07. Mô hình đề xuất có Sharpe top 1% cao nhất trên tập gộp ở cả hai giai đoạn (2,02 và 1,05). Ticker lọt vào top 5% và 10% của mô hình nhiều hơn của EB (Bảng 6), nên EB co nhiễu mạnh hơn ngoài top 1%.

**5.2.3 Summary of the noise analysis (Tóm tắt phân tích nhiễu).** Mô hình đề xuất hầu như không chọn ticker vào top 1%, như EB, và vẫn giữ Sharpe top 1% cao nhất khi thêm các chiến lược nhiễu vào tập. Ngoài top 1%, EB co các chiến lược nhiễu mạnh hơn.

#### 5.3 Temporal performance analysis (Phân tích theo thời gian)

**5.3.1 Cumulative returns (Lãi cộng dồn).** Hình 1 vẽ lãi cộng dồn của danh mục top 1%. Mô hình đề xuất nằm trên EB gần như cả mẫu và kết thúc ở 262% so với 241%, baseline kết thúc ở 120%. Naive mining đạt lãi cộng dồn tương đương (264%) nhưng dao động mạnh hơn, nên Sharpe holdout thấp hơn (Bảng 3). Mọi phương pháp lãi chủ yếu trước 2004 và Sharpe holdout đều thấp hơn, phù hợp với việc khả năng dự báo suy giảm mà Chen & Dim ghi nhận, và họ gắn điều này với sự phát triển của công nghệ thông tin quanh 2004 cùng sự chú ý hạn chế và việc thông tin lan truyền chậm.

**5.3.2 Year-by-year performance (Hiệu quả từng năm).** Mô hình đề xuất không áp đảo từng năm. Lãi năm lịch của mô hình cao hơn EB ở chỉ 21/38 năm (13/22 ở dev, 8/16 ở holdout), và cả hai đều dương ở 33 năm. Khác biệt nằm ở các năm xấu. Năm tệ nhất của mô hình là −2,5% (2005) so với −8,1% của EB (2020), và mức sụt giảm tối đa của lãi cộng dồn là 6,6 điểm phần trăm so với 10,3. Ở holdout, độ lệch chuẩn lãi năm là 3,5% so với 4,1%; ở dev hơi cao hơn (6,0% so với 5,5%), tương ứng với lãi cao hơn.

**5.3.3 Performance stability (Độ ổn định hiệu quả).** Bảng 7 chia mẫu thành bốn giai đoạn con. Sharpe top 1% của mô hình cao hơn EB ở cả bốn, và cao hơn naive mining ở ba giai đoạn cuối. Điều này gồm cả giai đoạn cuối, khi khả năng dự báo yếu nhất (0,99 so với 0,58).

#### 5.4 Discussion and limitations (Thảo luận và giới hạn)

**5.4.1 Interpretation (Diễn giải).** Lợi thế của mô hình tập trung ở một chỗ: đầu bảng xếp hạng, và vẫn còn sau 2004. Điều này phù hợp với thiết kế. Nhãn ở mục 3.1 so sánh các chiến lược trong cùng năm, và ablation ở mục 4.3 cho thấy nhãn này và bộ mã hoá pretrained có đóng góp, còn ISAB, Huber, trọng số năm thì không phân biệt được với dao động giữa các lần chạy. Chọn chiến lược có lãi trong mẫu lớn cũng chọn luôn sai số lấy mẫu lớn, và EB sửa điều này bằng cách co t-stat. Vì vậy dự báo của EB gần lãi thật hơn về độ lớn, có thể giải thích độ đúng mức tốt hơn và IC dev cao hơn. T-stat đã chứa phần lớn thông tin cho đầu bảng xếp hạng, nên naive mining tốt, phù hợp với Proposition 1 của Chen & Dim. Mô hình đề xuất vẫn bổ sung thêm. Hai chiến lược có cùng t-stat có thể khác nhau ở cách lợi suất phân bố theo thời gian, và các chiến lược chỉ mô hình chọn đạt Sharpe cao hơn nhiều các chiến lược chỉ EB chọn (mục 5.1), nên lịch sử lợi suất có thể chứa thêm thông tin.

**5.4.2 Practical implications (Ý nghĩa thực tiễn).** Với người khai thác dữ liệu giữ top 1% (khoảng 257 chiến lược mỗi năm) trong 12 tháng, mô hình cho lãi trên mỗi đơn vị rủi ro cao hơn và thua lỗ nhỏ hơn ở các năm xấu so với EB, chỉ dùng thông tin có sẵn theo thời gian thực. Vì mô hình xếp hạng tốt hơn nhưng ước lượng độ lớn kém hơn, nó phù hợp với việc chọn chiến lược để giữ, không phải để dự báo chúng sẽ lãi bao nhiêu. Lãi holdout của top 1% là 3,5%/năm trước chi phí giao dịch, thấp hơn lãi dev, nên lãi sau 2004 chỉ ở mức khiêm tốn.

**5.4.3 Limitations (Giới hạn).** Một, mô hình được chọn bằng dev sau khi thử khoảng 60 cấu hình, nên các $p$ chỉ mang tính mô tả, không hiệu chỉnh cho kiểm định nhiều lần (multiple testing). Hai, học lại cấu hình cuối cho Sharpe holdout thấp hơn trung bình (0,84 qua ba lần chạy, so với 0,96 của mô hình đã lưu), nên mô hình đã lưu có thể tốt hơn một lần học lại thông thường. Ba, bằng chứng của ta là về họ kế toán chia đều, nơi chiếm phần lớn top 1% của EB, và lợi thế so với EB chỉ có hỗ trợ thống kê ở top 1%; nó không mở rộng sang các thị trường khác. Bốn, FinCast được pretrained trên dữ liệu ngoài có thể trùng giai đoạn với các năm dự báo, và ta không loại trừ được việc điều này cho mô hình một ít thông tin về chúng. Cuối cùng, danh mục tính trước chi phí giao dịch.

---

## Cần bạn biết trước khi paste

1. **Phân tích mới 5.1.3 (nơi hai phương pháp bất đồng)** tính từ dự báo đã lưu của mô hình chốt: chỉ 51,5% top 1% của mô hình trùng với EB. Chiến lược chỉ mô hình chọn đạt Sharpe 1,63 so với 0,85 của chiến lược chỉ EB chọn (p < 0,001; holdout p = 0,038). Chênh lệch lãi trung bình riêng nó chưa có ý nghĩa (t = 0,99), bài ghi rõ điều đó.
2. **Số khác khớp** `docs/7_COMPARE_WITH_EB.md`, `table2.csv`, `extra.json` và `experiments/ticker_test/result.md`; số ở 5.3 (21/38 năm, năm tệ nhất −2,5% / −8,1%, sụt giảm tối đa 6,6 / 10,3 điểm, Sharpe 4 giai đoạn) tính bằng `evaluate.py`.
3. **Sự thật không đẹp nhưng đã ghi trong bài:** chỉ thắng EB ở 21/38 năm, IC dev và đúng mức EB hơn (ở 4.2.4), ticker lọt nhiều hơn ở top 5–10%, naive mining ngang ở dev. Lợi thế nằm ở top 1%, Sharpe, năm xấu nhẹ hơn, holdout và các chiến lược bất đồng với EB.
4. **Về "SOTA":** bài không viết "state-of-the-art". Câu "cao nhất trong bốn phương pháp được so sánh" nằm ở mục 6.2.
5. Chưa biên dịch LaTeX.
