import pandas as pd
from datetime import datetime, timedelta

def load_csv(file_path):
    """
    Load the CSV file and return a DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        data['Creacion Orden de Venta'] = pd.to_datetime(data['Creacion Orden de Venta'])
        print("CSV loaded successfully!")
        return data
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None


def group_by_week(data):
    """
    Group data into weekly intervals and aggregate quantities by article and unit,
    treating both "L" and "KG" as "KG".
    """
    start_date = datetime(2021, 1, 4)
    end_date = datetime(2023, 12, 31)
    weekly_data = []
    week_number = 1
    week_in_year = 1
    current_year = start_date.year

    # Normalize "Unidad de venta" to treat "L" as "KG"
    data['Unidad de venta'] = data['Unidad de venta'].replace('L', 'KG')

    while start_date <= end_date:
        week_end = start_date + timedelta(days=6)
        week_data = data[(data['Creacion Orden de Venta'] >= start_date) & (data['Creacion Orden de Venta'] <= week_end)]

        # Aggregate quantities by "Articulo" and normalized "Unidad de venta"
        aggregated = week_data.groupby(['Articulo', 'Unidad de venta'], as_index=False).agg({'Cantidad': 'sum'})

        # Reset week_in_year if the year changes
        if start_date.year != current_year:
            current_year = start_date.year
            week_in_year = 1  # Reset week number within the year

        week_label = f"{start_date.year}-Week{str(week_in_year).zfill(2)}"
        aggregated.insert(0, 'Week', week_label)  # Add week label as the first column
        aggregated['Week Start'] = start_date.strftime('%Y-%m-%d')
        aggregated['Week End'] = week_end.strftime('%Y-%m-%d')

        # Add a column with the cumulative week number
        aggregated['Week Number'] = week_number

        weekly_data.append(aggregated)

        start_date += timedelta(weeks=1)
        week_number += 1
        week_in_year += 1

    # Concatenate all weekly data into one DataFrame
    final_data = pd.concat(weekly_data, ignore_index=True)
    return final_data



def export_to_csv(data, output_path):
    """
    Export the final DataFrame to a CSV file.
    """
    try:
        data.to_csv(output_path, index=False)
        print(f"Data exported successfully to {output_path}")
    except Exception as e:
        print(f"Error exporting data: {e}")


def main():
    file_path = '/Users/juansalazar/Library/Mobile Documents/com~apple~CloudDocs/Documents/Estudio/Tecnológico de Monterrey/Datathon_2025_Interlub/utils/DatathonData.csv'  # Update the path if necessary
    output_path = 'weekly_aggregated_data.csv'  # Output CSV path
    data = load_csv(file_path)

    if data is not None:
        # Group data by week
        weekly_data = group_by_week(data)

        # Export the result to a CSV file
        export_to_csv(weekly_data, output_path)


if __name__ == "__main__":
    main()
