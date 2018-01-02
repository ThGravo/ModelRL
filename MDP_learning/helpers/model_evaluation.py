def sk_eval(model, input_data, train_signal, outfile_name=None):
    from sklearn.metrics import r2_score
    text_file = open(outfile_name, "w") if outfile_name is not None else None

    print(model.summary(), file=text_file)

    y_pred = model.predict(input_data)
    print('r2_score(uniform_average): '
          '{}'.format(r2_score(train_signal, y_pred, multioutput='uniform_average')),
          file=text_file)
    print('r2_score(variance_weighted): '
          '{}'.format(r2_score(train_signal, y_pred, multioutput='variance_weighted')),
          file=text_file)
    print('r2_score(raw_values): '
          '{}'.format(r2_score(train_signal, y_pred, multioutput='raw_values')),
          file=text_file)
