#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    #cleaned_data = []

    ### your code goes here
    residual_error = (net_worths - predictions) ** 2
    cleaned_data = zip(ages, net_worths, residual_error)
    cleaned_data.sort(key=lambda tup: tup[1], reverse=False)
    limit = int(len(net_worths)*0.1)
    return cleaned_data[limit:]

