# LaTeX Formatting Guidelines

## Two-Column Paper Format

### Tables
- **Always use `table*`** for tables that span the full width of the page in two-column papers
- **Never use `table`** alone for wide tables in two-column format
- Example:
  ```latex
  \begin{table*}[t]
  \centering
  \caption{Your caption here}
  \label{tab:your_label}
  % table content
  \end{table*}
  ```

### Figures
- **Use `figure*`** for wide figures that span both columns
- **Use `figure`** only for single-column figures
- Example:
  ```latex
  \begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{your_figure.pdf}
  \caption{Your caption here}
  \label{fig:your_label}
  \end{figure*}
  ```

## Float Placement

### Always Use `[t]` Placement
- **Preferred**: `[t]` - places float at top of page
- **Avoid**: `[htbp]` - can cause poor placement in two-column format
- **Reasoning**: Top placement ensures consistent layout and prevents floats from breaking column flow

Examples:
```latex
% GOOD
\begin{table*}[t]
\begin{figure*}[t]

% AVOID
\begin{table*}[htbp]
\begin{figure*}[h!]
```

## Caption and Label Conventions

### Order Matters
Always place elements in this order:
1. `\caption{}` - The caption text
2. `\label{}` - The reference label

```latex
\caption{Performance metrics for organ detection}
\label{tab:organ_metrics}
```

## Avoiding Package Dependencies

### Table Notes Without Extra Packages
Instead of using `tablenotes` or `threeparttable`, use `\parbox`:

```latex
\begin{table*}[t]
\centering
\caption{Your table caption}
\label{tab:example}
% ... table content ...
\end{table*}

\vspace{0.5em}
\parbox{\textwidth}{%
\footnotesize
\textbf{Note:} Your table notes here.
}
```

### Subfigures Without Subfigure Package
For side-by-side figures without `subfigure` or `subcaption`:

```latex
\begin{figure*}[t]
\centering
\begin{tabular}{cc}
\includegraphics[width=0.45\textwidth]{fig1.pdf} &
\includegraphics[width=0.45\textwidth]{fig2.pdf} \\
(a) First figure & (b) Second figure
\end{tabular}
\caption{Overall caption for both figures}
\label{fig:both}
\end{figure*}
```

## Table Formatting Best Practices

### Use Booktabs
Always use `\toprule`, `\midrule`, and `\bottomrule` from the booktabs package:

```latex
\begin{tabular}{lrr}
\toprule
Model & Accuracy & F1 Score \\
\midrule
GPT-4.1 & 85.2 & 82.3 \\
Claude & 83.1 & 80.5 \\
\bottomrule
\end{tabular}
```

### Column Alignment
- `l` - left align (for text)
- `r` - right align (for numbers)
- `c` - center (for short labels)

### Resizing Tables
For wide tables, use `\resizebox` carefully:

```latex
\resizebox{\textwidth}{!}{%
\begin{tabular}{...}
% table content
\end{tabular}%
}
```

## Common Issues and Solutions

### Issue: Table/Figure appears in wrong column
**Solution**: Use `*` versions (`table*`, `figure*`) and `[t]` placement

### Issue: Subfigure package not available
**Solution**: Use tabular environment or minipages for side-by-side content

### Issue: Table notes requiring special packages
**Solution**: Use `\parbox{\textwidth}{}` after the table

### Issue: Float placement breaking column flow
**Solution**: Always use `[t]` placement instead of `[htbp]`

## Checklist for Paper Submission

- [ ] All wide tables use `table*`
- [ ] All wide figures use `figure*`
- [ ] All floats use `[t]` placement
- [ ] No dependency on subfigure/subcaption packages
- [ ] No dependency on tablenotes package
- [ ] Tables use booktabs rules
- [ ] Captions appear before labels
- [ ] Font sizes are large enough for readability