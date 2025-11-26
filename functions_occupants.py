import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List


# ========================================================================
# Statitsical distributions of household sizes and number of occupants
# ========================================================================

# Household sizes (average values / in square meters) and number of occupants for each index
household_size = np.array([30, 50, 70, 90, 110, 130, 150, 170, 190, 210])
number_of_occupants = np.array([1, 2, 3, 4, 5, 7]) # 6+ represented as 6~8; 7 is average for calculating the area per occupant


'''
The statistical distributions are for the whole Germany, not region specific.

'''

# The statistical distribution of household areas (in 20 sqm intervals from 40 to 220 sqm)
area_pmf = np.array([0.053, 0.173, 0.228, 0.164, 0.115, 0.106, 0.067, 0.035, 0.022, 0.037])


# For number of occupants, the statistical probability distribution over household areas
# Rows: household area in 20 sqm intervals from 40 to 220 sqm
# Columns: number of occupants from 1 to 5, and 6+

area_of_occupants_pmf_ = np.array([
    [0.114, 0.009, 0.005, 0.003, 0.003, 0.006],
    [0.303, 0.103, 0.051, 0.026, 0.024, 0.033],
    [0.27,  0.227, 0.195, 0.132, 0.122, 0.123],
    [0.136, 0.196, 0.182, 0.164, 0.176, 0.175],
    [0.071, 0.152, 0.151, 0.145, 0.136, 0.143],
    [0.05,  0.137, 0.157, 0.175, 0.15, 0.137],
    [0.026, 0.08,  0.106, 0.136, 0.127, 0.107],
    [0.012, 0.038, 0.057, 0.081, 0.083, 0.071],
    [0.007, 0.023, 0.036, 0.052, 0.059, 0.056],
    [0.012, 0.035, 0.058, 0.086, 0.12, 0.149]
    ])

# For each household size (area), the statistical distribution over the number of occupants 
occupants_of_area_pmf = np.array([
    [0.927059110633631, 0.0529605807557342, 0.0119022574036725, 0.00474706475811662, 0.00158999086491576, 0.00174380062515019],
    [0.761194877739394, 0.181579795144092, 0.0364279365700298, 0.0136359191046399, 0.00415896036058322, 0.00300179198582519],
    [0.512811008795322, 0.303789200431076, 0.106266355674648, 0.0526615859308795, 0.0159691564637147, 0.00850214858077879],
    [0.358293329104416, 0.363701087755891, 0.138184977214084, 0.0910701239682177, 0.031944014860933, 0.0168073750224712],
    [0.266522381688543, 0.401267809660991, 0.162788604056836, 0.114767723296762, 0.0350901526256233, 0.0195637594155317],
    [0.204813755471113, 0.395424812742793, 0.185501976489373, 0.15127721131892, 0.0424552507968956, 0.0205258156477669],
    [0.168862937598044, 0.365832764140915, 0.197957376938244, 0.185682898906593, 0.0565052837292521, 0.0251583668682791],
    [0.14553198225371, 0.335057478640706, 0.204302662371277, 0.212448500250279, 0.0705366527052253, 0.0321241498708674],
    [0.136964020398921, 0.31973480948938, 0.20596871605524, 0.216531388935867, 0.0807784551438265, 0.0400248800547543],
    [0.137652389308018, 0.292968865816548, 0.196771219950028, 0.212200847962437, 0.0966134062160552, 0.0637966399555804]])
# Round to 3 decimal places
occupants_of_area_pmf = np.round(occupants_of_area_pmf, 3)


'''
# ============================================================================
# FUNCTIONS FOR CALCULATING NUMBER OF OCCUPANTS PER BUILDING
# ============================================================================
'''



# ============================================================================
# BUILDING CLASS
# ============================================================================

@dataclass
class Building:
    """
    Data class for building information required for occupant estimation.
    
    Attributes:
        building_id: Unique identifier for the building (gml_id)
        measured_height: Total height of the building in meters (citygml_measured_height)
        storeys_above_ground: Number of floors above ground (citygml_storeys_above_ground)
        volume: Building volume in cubic meters (_volume)
        function: Building function code (citygml_function)
        roof_type: Roof type code (citygml_roof_type)
        total_occupants: Total number of occupants in the building (to be calculated)
    """
    building_id: str
    measured_height: Optional[float] = None  # Changed from float
    storeys_above_ground: Optional[int] = None
    volume: Optional[float] = None
    function: Optional[str] = None
    roof_type: Optional[str] = None
    total_occupants: Optional[int] = None
    
    def validate(self) -> bool:
        """Check if building has required attributes with valid values"""
        # Check for None values
        if self.measured_height is None or self.storeys_above_ground is None:
            return False
        # Check for valid ranges
        return self.measured_height > 0 and self.storeys_above_ground > 0
    
    def get_height(self) -> Optional[float]:
        """Safely get building height as float"""
        return self.measured_height
    
    def get_storeys(self) -> int:
        """Safely get number of storeys as int (None becomes 0)"""
        return self.storeys_above_ground if self.storeys_above_ground is not None else 0
    
    def get_volume(self) -> Optional[float]:
        """Safely get building volume as float"""
        return self.volume
    
    def get_function(self) -> Optional[str]:
        """Get building function"""
        return self.function
    
    def get_roof_type(self) -> Optional[str]:
        """Get roof type"""
        return self.roof_type

# ============================================================================
# STEP 1: Calculate heated area per building
# Need to make adjustment to calculate the reduced area for service areas in the next step
# ============================================================================

def av_storey_h_and_h_area_building(building: Building) -> Tuple[float, float]:
    """
    Calculate average storey height and heated area of a building.
    
    Args:
        building: Building object with measured_height, storeys_above_ground and volume attributes
        
    Returns:
        tuple: (average_storey_height, heated_area)
               Returns (0, 0) if validation fails
    """
    # Validate building has required attributes
    if not building.validate():
        print("The building does not have a measured height or storeys above ground")
        return 0.0, 0.0
    
    # Get values (they're already proper types from Building class)
    measured_height = building.measured_height
    storeys_above_ground = building.storeys_above_ground
    volume = building.volume if building.volume is not None else 0.0
        
    # Calculate average storey height
    h_g = measured_height / storeys_above_ground
    
    # Calculate heated area based on storey height
    if 2.5 <= h_g <= 3.0:
        A_h = 0.32 * volume
    else:
        A_h = ((1 / h_g) - 0.04) * volume
    
    return h_g, A_h


# ============================================================================
# STEP 1.5: Calculate number of households per building
# ============================================================================

def calculate_number_of_households(
    heated_area: float,
    storeys: Optional[int] = None,
    measured_height: Optional[float] = None
) -> Tuple[str, int, float]:
    """
    Calculate the number of households in a building based on heated area.
    For calculation of number of households, the heated area is adjusted first, 
    and then the number of households is calculated based on the adjusted area
    
    Args:
        heated_area: Total heated area of the building in square meters
        storeys: Number of storeys above ground
        measured_height: Building height in meters
        
    Returns:
        tuple containing:
            - building_type: str - Residential building type classification
            - number_of_households: int - Estimated number of households
            - adjusted_heated_area: float - Heated area after accounting for service areas
    
    Building Type Classification:
        - SFH: heated_area ≤ 130.8 m²
        - MFH: 130.8 < heated_area < 1000 m² and storeys < 5
        - AB: heated_area ≥ 1000 m² or storeys ≥ 5
        - HRB: measured_height > 22 m or storeys > 8
    """
    # Determine building type with enhanced logic
    if measured_height and measured_height > 22.0:
        building_type = "HRB"  # High Rise Building
    elif storeys and storeys > 8:
        building_type = "HRB"
    elif heated_area >= 1000 or (storeys and storeys >= 5):
        building_type = "AB"  # Apartment Block
    elif heated_area > 130.8:
        building_type = "MFH"  # Multi-Family Home
    else:
        building_type = "SFH"  # Single-Family Home
    
    # Calculate households and adjust area
    household_area_map = {
        "SFH": (1, 1.0),        # 1 household, no reduction
        "MFH": (80.2, 0.59),    # 80.2 m²/household, 59% usable
        "AB": (62.4, 0.59),     # 62.4 m²/household, 59% usable
        "HRB": (54.3, 0.59)     # 54.3 m²/household, 59% usable
    }
    
    if building_type == "SFH":
        number_of_households = 1
        adjusted_heated_area = heated_area
    else:
        area_per_household, reduction_factor = household_area_map[building_type]
        adjusted_heated_area = heated_area * reduction_factor
        number_of_households = max(1, round(adjusted_heated_area / area_per_household))  # At least 1 household
    
    return building_type, number_of_households, adjusted_heated_area


# ============================================================================
# STEP 2: Calculate household areas
# ============================================================================

def getNewHouseholdSize(area_pmf: np.ndarray) -> float:
    """
    Get new household size based on statistical distribution.
    
    Args:
        area_pmf: Probability mass function for household areas
        
    Returns:
        float: Randomly generated household area in square meters
    """
    # Generate random integer [0, 999]
    n = random.randint(0, 999)
    tempSum = 0
    
    for i in range(len(area_pmf)):
        tempSum += area_pmf[i] * 1000
        
        if n < tempSum:
            # Return area in range [20+i*20, 20+i*20+20)
            min_area = 20 + i * 20
            area = min_area + random.randint(0, 19)
            return float(area)
    
    # Fallback (should rarely happen)
    return 100.0


def isValid(area: float, remaining_area: float, remaining_households: int) -> bool:
    """
    Validate if new area value can be placed in the building.
    
    Args:
        area: New household area to validate
        remaining_area: Remaining area in the building
        remaining_households: Remaining number of households to allocate
        
    Returns:
        bool: True if valid, False otherwise
    """
    # Fixed: Check for remaining_households <= 1 (not <= 0)
    if area is None or remaining_households <= 1:
        return False
    
    # Fixed: Now safe from division by zero since remaining_households > 1
    avg_remaining = (remaining_area - area) / (remaining_households - 1)
    return 40 <= avg_remaining <= 160


def heated_area_per_household(
    A_h: float, 
    number_of_households: int,
    area_pmf: np.ndarray,
    timeout: float = 500
) -> List[float]:
    """
    Distribute heated area for each household in a building using statistical distribution.
    
    Args:
        A_h: Total heated area of the building
        number_of_households: Number of households in the building
        area_pmf: Probability mass function for household areas
        timeout: Maximum time allowed for calculation (seconds)
        
    Returns:
        list: Sorted list of household areas, or empty list if calculation fails
    """
    if number_of_households <= 0:
        return [A_h] if A_h > 0 else []
    
    if number_of_households == 1:
        return [A_h]
    
    start_time = time.time()
    A_h_per_household = []
    remaining_area = A_h
    remaining_households = number_of_households
    
    for household in range(number_of_households):
        # Check timeout
        if time.time() - start_time > timeout:
            print(f"Time limit exceeded for heated area per household calculation")
            return []
        
        # Last household gets remaining area
        if remaining_households == 1:
            A_h_per_household.append(remaining_area)
            return sorted(A_h_per_household)
        
        # Try to find valid area
        attempts = 0
        max_attempts = 10000  # Prevent infinite loop
        
        while attempts < max_attempts:
            area = getNewHouseholdSize(area_pmf)
            
            # Fixed: Now passing remaining_households (not -1)
            if isValid(area, remaining_area, remaining_households):
                A_h_per_household.append(area)
                remaining_area -= area
                remaining_households -= 1
                break
            
            attempts += 1
        
        # If no valid area found after max attempts, distribute evenly
        if attempts >= max_attempts:
            print(f"Warning: Could not find valid distribution, using average")
            avg_area = remaining_area / remaining_households
            for _ in range(remaining_households):
                A_h_per_household.append(avg_area)
            return sorted(A_h_per_household)
    
    return sorted(A_h_per_household)


# ============================================================================
# STEP 3: Calculate number of occupants per household
# ============================================================================

def number_occupants_per_household(
    area: float,
    occupants_of_area_pmf: np.ndarray
) -> int:
    """
    Get number of occupants for a household based on heated area.
    
    Args:
        area: Household area in square meters
        occupants_of_area_pmf: Statistical distribution table [10 x 6]
                               Rows: area bins, Columns: number of occupants (1-5, 6+)
        
    Returns:
        int: Number of occupants (1-8)
    """
    # Calculate index based on area (20 sqm bins starting from 20)
    index = int((area - 20) / 20)
    
    # Clamp index to valid range [0, 9]
    index = max(0, min(9, index))
    
    # Generate random number [0, 999]
    n = random.randint(0, 999)
    tempSum = 0
    
    # Iterate through occupant categories (1-5, 6+)
    for i in range(6):
        tempSum += occupants_of_area_pmf[index][i] * 1000
        
        if n < tempSum:
            number_of_occupants = i + 1
            
            # For 6+ category, add random 0-2
            if number_of_occupants == 6:
                number_of_occupants = 6 + random.randint(0, 2)
            
            return number_of_occupants
    
    # Fallback
    return 1

# ============================================================================
# STEP 4: Calculate total number of occupants per building
# ============================================================================

def calculate_building_occupants(
    household_areas: List[float],
    occupants_of_area_pmf: np.ndarray,
    verbose: bool = False
) -> Tuple[int, List[int], dict]:
    """
    Calculate the total number of occupants in a building.
    
    Args:
        household_areas: List of household areas in square meters
        occupants_of_area_pmf: Statistical distribution table [10 x 6]
        verbose: If True, print details for each household
        
    Returns:
        tuple containing:
            - total_occupants: int - Total number of occupants in the building
            - occupants_per_household: List[int] - Number of occupants for each household
            - occupants_by_size: dict - Distribution of households by number of occupants
                                       Keys: household size (1p, 2p, 3p, 4p, 5p+)
                                       Values: count of households
    
    Example:
        >>> household_areas = [85.2, 92.1, 78.5]
        >>> total, per_household, distribution = calculate_building_occupants(
        ...     household_areas, occupants_of_area_pmf, verbose=True
        ... )
        >>> print(f"Total occupants: {total}")
    """
    if not household_areas:
        return 0, [], {"1p": 0, "2p": 0, "3p": 0, "4p": 0, "5p+": 0}
    
    total_occupants = 0
    occupants_per_household = []
    
    # Initialize distribution counter
    occupants_by_size = {
        "1p": 0,   # 1 person households
        "2p": 0,   # 2 person households
        "3p": 0,   # 3 person households
        "4p": 0,   # 4 person households
        "5p+": 0   # 5+ person households
    }
    
    # Calculate occupants for each household
    for i, area in enumerate(household_areas):
        n_occ = number_occupants_per_household(area, occupants_of_area_pmf)
        occupants_per_household.append(n_occ)
        total_occupants += n_occ
        
        # Update distribution
        if n_occ == 1:
            occupants_by_size["1p"] += 1
        elif n_occ == 2:
            occupants_by_size["2p"] += 1
        elif n_occ == 3:
            occupants_by_size["3p"] += 1
        elif n_occ == 4:
            occupants_by_size["4p"] += 1
        else:  # n_occ >= 5
            occupants_by_size["5p+"] += 1
        
        # Print details if verbose
        if verbose:
            print(f"Household {i+1}: {area:.1f}m² → {n_occ} occupants")
    
    return total_occupants, occupants_per_household, occupants_by_size


def calculate_building_occupants_summary(
    household_areas: List[float],
    occupants_of_area_pmf: np.ndarray
) -> dict:
    """
    Calculate building occupants with detailed summary statistics.
    
    Args:
        household_areas: List of household areas in square meters
        occupants_of_area_pmf: Statistical distribution table [10 x 6]
        
    Returns:
        dict containing:
            - total_occupants: Total number of occupants
            - total_households: Total number of households
            - avg_occupants_per_household: Average occupants per household
            - occupants_per_household: List of occupants for each household
            - distribution: Household distribution by size
            - area_stats: Statistics about household areas
    """
    total_occ, occ_per_hh, distribution = calculate_building_occupants(
        household_areas, occupants_of_area_pmf, verbose=False
    )
    
    summary = {
        "total_occupants": total_occ,
        "total_households": len(household_areas),
        "avg_occupants_per_household": total_occ / len(household_areas) if household_areas else 0,
        "occupants_per_household": occ_per_hh,
        "distribution": distribution,
        "area_stats": {
            "min_area": min(household_areas) if household_areas else 0,
            "max_area": max(household_areas) if household_areas else 0,
            "avg_area": np.mean(household_areas) if household_areas else 0,
            "total_area": sum(household_areas)
        }
    }
    
    return summary  # Fixed: return only summary, not tuple


def print_building_occupants_report(summary: dict):
    """
    Print a formatted report of building occupants.
    
    Args:
        summary: Dictionary returned by calculate_building_occupants_summary()
    """
    print("\n" + "="*70)
    print("BUILDING OCCUPANTS REPORT")
    print("="*70)
    
    # Fixed: Check for zero households to prevent division by zero
    if summary['total_households'] == 0:
        print("\nNo households in building.")
        print("="*70 + "\n")
        return
    
    print(f"\nTotal Occupants:      {summary['total_occupants']} persons")
    print(f"Total Households:     {summary['total_households']}")
    print(f"Avg Occupants/HH:     {summary['avg_occupants_per_household']:.2f} persons/household")
    
    print("\n" + "-"*70)
    print("HOUSEHOLD DISTRIBUTION BY SIZE")
    print("-"*70)
    dist = summary['distribution']
    total_hh = summary['total_households']
    print(f"1-person households:  {dist['1p']:3d} ({dist['1p']/total_hh*100:5.1f}%)")
    print(f"2-person households:  {dist['2p']:3d} ({dist['2p']/total_hh*100:5.1f}%)")
    print(f"3-person households:  {dist['3p']:3d} ({dist['3p']/total_hh*100:5.1f}%)")
    print(f"4-person households:  {dist['4p']:3d} ({dist['4p']/total_hh*100:5.1f}%)")
    print(f"5+ person households: {dist['5p+']:3d} ({dist['5p+']/total_hh*100:5.1f}%)")
    
    print("\n" + "-"*70)
    print("HOUSEHOLD AREA STATISTICS")
    print("-"*70)
    area = summary['area_stats']
    print(f"Total Area:       {area['total_area']:.2f} m²")
    print(f"Average Area:     {area['avg_area']:.2f} m²/household")
    print(f"Min Area:         {area['min_area']:.2f} m²")
    print(f"Max Area:         {area['max_area']:.2f} m²")
    
    # Fixed: Check for zero occupants to prevent division by zero
    if summary['total_occupants'] > 0:
        print(f"Area per Person:  {area['total_area']/summary['total_occupants']:.2f} m²/person")
    else:
        print(f"Area per Person:  N/A (no occupants)")
    
    print("="*70 + "\n")


# ============================================================================
# TESTING EXAMPLES
# ============================================================================

# Example 1: Test Building class
building_test = Building(
    building_id="BLD_001",
    measured_height=15.0,
    storeys_above_ground=5,
    volume=3000.0,
)

print("Building validation:", building_test.validate())

# Example 2: Test heated area calculation
h_g, A_h = av_storey_h_and_h_area_building(building_test)
print(f"\nAverage storey height: {h_g:.2f} m")
print(f"Heated area: {A_h:.2f} m²")

# Example 3: Test household calculation
building_type, n_households, adjusted_A_h = calculate_number_of_households(
    heated_area=A_h,
    storeys=5,
    measured_height=15.0
)
print(f"\nBuilding type: {building_type}")
print(f"Number of households: {n_households}")
print(f"Adjusted heated area: {adjusted_A_h:.2f} m²")

# Example 4: Test area distribution (requires area_pmf)
household_areas = heated_area_per_household(adjusted_A_h, n_households, area_pmf)
print(f"\nHousehold areas: {[f'{a:.1f}' for a in household_areas]} m²")

# Example 5: Test occupants calculation (requires occupants_of_area_pmf)
# This should be defined from your earlier cells
if 'occupants_of_area_pmf' in globals():
    for area in household_areas[:3]:  # Test first 3 households
        n_occ = number_occupants_per_household(area, occupants_of_area_pmf)
        print(f"Household {area:.1f}m² → {n_occ} occupants")



# Example 6: Full building occupants calculation
if 'occupants_of_area_pmf' in globals() and household_areas:
    print("\n" + "="*70)
    print("EXAMPLE: Calculate Building Occupants")
    print("="*70)
    
    # Method 1: Simple calculation with verbose output
    print("\n--- Method 1: Simple Calculation ---")
    total_occ, occ_list, distribution = calculate_building_occupants(
        household_areas, 
        occupants_of_area_pmf, 
        verbose=True
    )
    
    print(f"\n>>> Total Occupants in Building: {total_occ} persons")
    print(f">>> Total Households: {len(household_areas)}")
    print(f">>> Average: {total_occ/len(household_areas):.2f} persons/household")
    print(f">>> Distribution: {distribution}")
    
    # Method 2: Detailed summary
    print("\n--- Method 2: Detailed Summary ---")
    summary = calculate_building_occupants_summary(household_areas, occupants_of_area_pmf)
    print_building_occupants_report(summary)
    
    # Method 3: Quick one-liner for total only
    print("\n--- Method 3: Quick Calculation ---")
    total = sum(number_occupants_per_household(area, occupants_of_area_pmf) 
                for area in household_areas)
    print(f"Total occupants (quick method): {total} persons")