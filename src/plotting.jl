function plot_XNOR!(ax)
    x = [b[1] for b in X_BITS]
    y = [b[2] for b in X_BITS]

    # Split by class
    x_class1 = [x[i] for i in eachindex(x) if y_XNOR[i] == 1.0]
    y_class1 = [y[i] for i in eachindex(y) if y_XNOR[i] == 1.0]

    x_class0 = [x[i] for i in eachindex(x) if y_XNOR[i] == 0.0]
    y_class0 = [y[i] for i in eachindex(y) if y_XNOR[i] == 0.0]

    # Scatter classes
    scatter!(ax, x_class1, y_class1, markersize = 18, label = L"\text{XNOR} = 1",color = :grey)
    scatter!(ax, x_class0, y_class0, markersize = 18, marker = :rect, label = L"\text{XNOR} = 0",color = :yellow)

    # Annotate points
    for i in eachindex(X_BITS)
        b = X_BITS[i]
        text!(
            ax,
            "($(b[1]), $(b[2]))",
            position = (b[1], b[2] + 0.05),align = (:center,:bottom),
            fontsize = 18,color = colors[i]
        )
    end

    # Optional: diagonal guides to emphasize structure
    lines!(ax, [-0.1, 1.1], [-0.1, 1.1], linestyle = :dash,color = :grey)
    lines!(ax, [-0.1, 1.1], [1.1, -0.1], linestyle = :dash,color = :yellow)

end