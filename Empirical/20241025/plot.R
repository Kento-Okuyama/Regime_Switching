# plot
ggplot(df_combined, aes(x = Time)) +
  geom_line(aes(y = Predicted_Value, color = "Predicted"), size = 1, alpha = 0.7) +
  geom_line(aes(y = True_Value, color = "True", linetype = "True"), size = 1, alpha = 0.7) +
  facet_wrap(~ ID, ncol = 5, scales = "free_y") +
  labs(title = "Comparison of Predicted and True Values for Selected Individuals",
       x = "Time",
       y = "Value",
       color = "Legend") +
  scale_color_manual(values = c("Predicted" = "blue", "True" = "red")) +
  scale_linetype_manual(values = c("True" = "dashed")) +
  theme_minimal(base_size = 15) +
  coord_cartesian(ylim = c(0, 1)) +
  theme(
    strip.text = element_text(size = 12),
    panel.grid = element_blank(),
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank()
  ) +
  # 20, 24, 38番目のプロットにのみ y 軸を表示
  facet_wrap(~ ID, ncol = 5, labeller = as_labeller(function(x) {
    if (x %in% c("20", "24", "38")) {
      return(element_text())
    } else {
      return(element_blank())
    }
  }))