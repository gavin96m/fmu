from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(Y_test_orig, Y_pred_orig)
r2 = r2_score(Y_test_orig, Y_pred_orig)
mae = mean_absolute_error(Y_test_orig, Y_pred_orig)
rmse = np.sqrt(mse)


print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R^2 Score: {r2}")


mse_per_output = mean_squared_error(Y_test_orig, Y_pred_orig, multioutput='raw_values')
mae_per_output = mean_absolute_error(Y_test_orig, Y_pred_orig, multioutput='raw_values')


for i in range(Y_pred_orig.shape[1]):
    print(f"输出变量 {i}: MSE = {mse_per_output[i]}, MAE = {mae_per_output[i]}")

