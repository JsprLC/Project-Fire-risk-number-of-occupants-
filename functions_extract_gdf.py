
# ============================================================================
# Function to Convert GeoDataFrame Row to Building Object
# ============================================================================
from functions_occupants import Building

def extract_building_from_geodataframe(
    building_series: gpd.GeoSeries,
    building_id_col: str = 'gml_id',
    height_col: str = 'citygml_measured_height',
    storeys_col: str = 'citygml_storeys_above_ground',
    volume_col: str = '_volume',
    function_col: str = 'citygml_function',
    roof_type_col: str = 'citygml_roof_type'
) -> Building:
    """
    Convert a GeoDataFrame row (pandas Series) to a Building object.
    For storeys, if None or NaN, estimate from height assuming 3m per storey
    
    Args:
        building_series: A row from GeoDataFrame (pandas.Series)
        building_id_col: Column name for building ID (gml_id)
        height_col: Column name for measured height (citygml_measured_height)
        storeys_col: Column name for storeys above ground (citygml_storeys_above_ground)
        volume_col: Column name for volume (_volume)
        function_col: Column name for building function (citygml_function)
        roof_type_col: Column name for roof type (citygml_roof_type)
        
    Returns:
        Building object without total_occupants (need to be adjusted later)
        
    Example:
        >>> building = extract_building_from_geodataframe(building_test)
        >>> print(building.building_id, building.volume)
    """
    # Helper function to safely convert to string (only for string fields!)
    def safe_str(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None  # Changed: return None instead of ''
        return str(value)
    
    # Helper function to safely get float value
    def safe_float(value):
        try:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return None
            return float(value)
        except (ValueError, TypeError):
            return None
    
    # Get height first (needed for storey estimation)
    measured_height = safe_float(building_series.get(height_col))
    
    # Helper function to safely get storeys with estimation fallback
    def safe_storeys(value, height):
        # If value exists and is valid, use it
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            try:
                return int(float(value))
            except (ValueError, TypeError):
                pass
        
        # Estimate from height if available
        if height is not None and height > 0:
            estimated_storeys = max(1, round(height / 3.0))
            return int(estimated_storeys)
        
        # Default to None if can't estimate
        return None
    
    # Extract values with correct types
    building_id = str(building_series.get(building_id_col, ''))  # ID is always string
    storeys_above_ground = safe_storeys(building_series.get(storeys_col), measured_height)
    volume = safe_float(building_series.get(volume_col))
    function = safe_str(building_series.get(function_col))  # Returns None or string
    roof_type = safe_str(building_series.get(roof_type_col))  # Returns None or string
    
    # Create Building object
    building = Building(
        building_id=building_id,
        measured_height=measured_height,
        storeys_above_ground=storeys_above_ground,
        volume=volume,
        function=function,
        roof_type=roof_type
    )
    
    return building


# ============================================================================
# Function to Extract Multiple Buildings
# ============================================================================

def extract_buildings_from_geodataframe(
    gdf: gpd.GeoDataFrame,
    filter_invalid: bool = True,
    **kwargs
) -> list[Building]:
    """
    Convert entire GeoDataFrame to list of Building objects.
    
    Args:
        gdf: GeoDataFrame containing building data
        filter_invalid: If True, only return valid buildings
        **kwargs: Additional arguments for extract_building_from_geodataframe
        
    Returns:
        List of Building objects
    """
    buildings = []
    
    for idx, row in gdf.iterrows():
        building = extract_building_from_geodataframe(row, **kwargs)
        
        if filter_invalid:
            if building.validate():
                buildings.append(building)
        else:
            buildings.append(building)
    
    return buildings


# ============================================================================
# Export Functions: convert Building to dict and print info
# ============================================================================

def building_to_dict(building: Building) -> dict:
    """Convert Building object to dictionary for export"""
    return {
        'building_id': building.building_id,
        'measured_height': building.measured_height,
        'storeys_above_ground': building.storeys_above_ground,
        'volume': building.volume,
        'function': building.function,
        'roof_type': building.roof_type,
        'is_valid': building.validate()
    }


def print_building_info(building: Building):
    """Print formatted building information"""
    print("="*70)
    print("BUILDING INFORMATION")
    print("="*70)
    print(f"Building ID:          {building.building_id}")
    print(f"Function:             {building.function if building.function else 'N/A'}")
    print(f"Roof Type:            {building.roof_type if building.roof_type else 'N/A'}")
    print(f"Measured Height:      {building.measured_height} m" if building.measured_height else "Measured Height: N/A")
    print(f"Storeys Above Ground: {building.storeys_above_ground}" if building.storeys_above_ground is not None else "Storeys Above Ground: N/A")
    print(f"Volume:               {building.volume:.2f} m³" if building.volume else "Volume: N/A")
    print(f"Valid:                {building.validate()}")
    print("="*70)

