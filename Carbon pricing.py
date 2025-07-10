import random
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np

class TransportMode(Enum):
    WALKING = "walking"
    PUBLIC_TRANSIT = "public_transit"
    ICE_CAR = "ice_car"
    EV_CAR = "ev_car"

class IncomeLevel(Enum):
    LOW = "low"      # <$2,000/month
    MIDDLE = "middle" # $2,000-$5,000/month
    HIGH = "high"    # >$5,000/month

@dataclass
class CarbonPricing:
    oil_price_per_barrel: float  # $/barrel
    carbon_price_per_gallon: float  # $/gallon
    
    def calculate_gas_price_per_gallon(self) -> float:
        """Calculate gas price per gallon including carbon pricing"""
        base_price = self.oil_price_per_barrel / 42  # 1 barrel = 42 gallons
        return base_price + self.carbon_price_per_gallon

@dataclass
class TransportCosts:
    # Cost per km for different modes
    walking_cost_per_km: float = 0.0
    public_transit_cost_per_trip: float = 2.0
    ice_fuel_consumption_l_per_km: float = 0.08
    ev_consumption_wh_per_km: float = 166.0
    electricity_cost_per_kwh: float = 0.15
    
    # Travel speeds (km/h)
    walking_speed: float = 5.0  # 5 km/h
    public_transit_speed: float = 25.0  # 25 km/h (including stops)
    car_speed: float = 50.0  # 50 km/h (urban driving)
    
    def get_ice_cost_per_km(self, gas_price_per_gallon: float) -> float:
        """Calculate ICE car cost per km including fuel"""
        fuel_cost = self.ice_fuel_consumption_l_per_km * gas_price_per_gallon / 3.785
        return fuel_cost
    
    def get_ev_cost_per_km(self) -> float:
        """Calculate EV cost per km"""
        return (self.ev_consumption_wh_per_km / 1000) * self.electricity_cost_per_kwh
    
    def get_travel_time(self, mode: TransportMode, distance: float) -> float:
        """Calculate travel time in hours for a given mode and distance"""
        if mode == TransportMode.WALKING:
            return distance / self.walking_speed
        elif mode == TransportMode.PUBLIC_TRANSIT:
            return distance / self.public_transit_speed
        elif mode in [TransportMode.ICE_CAR, TransportMode.EV_CAR]:
            return distance / self.car_speed
        return 0.0

@dataclass
class Emissions:
    walking_g_co2_per_km: float = 0.0
    public_transit_g_co2_per_km: float = 50.0
    ice_g_co2_per_km: float = 200.0
    ev_g_co2_per_km: float = 50.0  # Based on California grid

class Agent:
    def __init__(self, name: str, income_level: IncomeLevel, initial_mode: TransportMode):
        self.name = name
        self.income_level = income_level
        self.current_mode = initial_mode
        self.preferences = self._generate_preferences()
        self.vehicle_owned = self._get_initial_vehicle(initial_mode)
        self.monthly_income = self._get_income_range(income_level)
        self.decision_history = []
        self.months_with_current_mode = 0  # For Markov chain inertia
        
    def _generate_preferences(self) -> Dict[str, float]:
        """Generate random preferences for convenience vs cost vs eco-friendliness"""
        return {
            'convenience': random.uniform(0.2, 0.8),
            'cost_sensitivity': random.uniform(0.3, 0.9),
            'eco_friendliness': random.uniform(0.1, 0.7),
            'time_sensitivity': random.uniform(0.4, 0.9)  # New parameter for time sensitivity
        }
    
    def _get_initial_vehicle(self, mode: TransportMode) -> str:
        if mode == TransportMode.ICE_CAR:
            return "ICE"
        elif mode == TransportMode.EV_CAR:
            return "EV"
        else:
            return "None"
    
    def _get_income_range(self, level: IncomeLevel) -> float:
        if level == IncomeLevel.LOW:
            return random.uniform(1000, 2000)
        elif level == IncomeLevel.MIDDLE:
            return random.uniform(2000, 5000)
        else:  # HIGH
            return random.uniform(5000, 10000)
    
    def get_transit_budget(self) -> float:
        """Get monthly transit budget based on income level"""
        if self.income_level == IncomeLevel.LOW:
            return self.monthly_income * 0.15  # 15% of income
        elif self.income_level == IncomeLevel.MIDDLE:
            return self.monthly_income * 0.25  # 25% of income - allows EV adoption
        else:  # HIGH
            return self.monthly_income * 0.35  # 35% of income
    
    def calculate_mode_cost(self, mode: TransportMode, costs: TransportCosts, 
                          gas_price_per_gallon: float, daily_distance: float) -> float:
        """Calculate monthly cost for a transport mode"""
        monthly_distance = daily_distance * 30
        
        if mode == TransportMode.WALKING:
            return 0.0
        elif mode == TransportMode.PUBLIC_TRANSIT:
            # Assume 2 trips per day
            return (costs.public_transit_cost_per_trip * 2 * 30)
        elif mode == TransportMode.ICE_CAR:
            fuel_cost = costs.get_ice_cost_per_km(gas_price_per_gallon) * monthly_distance
            # Add maintenance and insurance
            return fuel_cost + 200  # $200/month for maintenance/insurance
        elif mode == TransportMode.EV_CAR:
            electricity_cost = costs.get_ev_cost_per_km() * monthly_distance
            # Add maintenance (lower for EV)
            return electricity_cost + 100  # $100/month for maintenance/insurance
    
    def can_afford_mode(self, mode: TransportMode, costs: TransportCosts, 
                       gas_price_per_gallon: float, daily_distance: float) -> bool:
        """Check if agent can afford a transport mode"""
        if self.income_level == IncomeLevel.LOW:
            # Low income can only walk or use public transit
            return mode in [TransportMode.WALKING, TransportMode.PUBLIC_TRANSIT]
        
        monthly_cost = self.calculate_mode_cost(mode, costs, gas_price_per_gallon, daily_distance)
        transit_budget = self.get_transit_budget()
        return monthly_cost <= transit_budget
    
    def can_switch_to_ev(self) -> bool:
        """Check if agent can afford to buy an EV"""
        if self.income_level == IncomeLevel.LOW:
            return False
        # Middle and high income can afford EVs
        return self.monthly_income >= 2500
    
    def should_reconsider_decision(self) -> bool:
        """Markov chain: probability of reconsidering decision based on time with current mode"""
        # Higher probability to change if recently switched, lower if long-term user
        if self.months_with_current_mode <= 1:
            return random.random() < 0.8  # 80% chance to reconsider if just switched
        elif self.months_with_current_mode <= 3:
            return random.random() < 0.4  # 40% chance after 1-3 months
        elif self.months_with_current_mode <= 6:
            return random.random() < 0.2  # 20% chance after 3-6 months
        else:
            return random.random() < 0.1  # 10% chance after 6+ months
    
    def make_decision(self, costs: TransportCosts, carbon_pricing: CarbonPricing, 
                     emissions: Emissions, daily_distance: float) -> Tuple[TransportMode, str]:
        """Agent makes transportation decision for the month using Markov chain approach"""
        
        # Check if agent should reconsider their decision (Markov chain)
        if not self.should_reconsider_decision():
            self.months_with_current_mode += 1
            return self.current_mode, f"maintain {self.current_mode.value.replace('_', ' ')} due to inertia"
        
        gas_price = carbon_pricing.calculate_gas_price_per_gallon()
        
        # Calculate costs for all available modes
        available_modes = []
        for mode in TransportMode:
            if self.can_afford_mode(mode, costs, gas_price, daily_distance):
                available_modes.append(mode)
        
        if not available_modes:
            # If can't afford any mode, default to walking
            return TransportMode.WALKING, "Forced to walk due to affordability"
        
        # Calculate utility scores for each mode
        mode_scores: Dict[TransportMode, float] = {}
        for mode in available_modes:
            cost = self.calculate_mode_cost(mode, costs, gas_price, daily_distance)
            emission = self._get_emission_for_mode(mode, emissions, daily_distance)
            travel_time = costs.get_travel_time(mode, daily_distance)
            
            # Utility function based on preferences
            cost_utility = max(0, 1 - (cost / self.get_transit_budget()))  # Bounded utility
            emission_utility = max(0, 1 - (emission / (emissions.ice_g_co2_per_km * daily_distance * 30)))
            time_utility = max(0, 1 - (travel_time / 2.0))  # Normalize to 2 hours max
            
            # Weighted utility score
            utility = (self.preferences['cost_sensitivity'] * cost_utility + 
                      self.preferences['eco_friendliness'] * emission_utility + 
                      self.preferences['convenience'] * self._get_convenience_score(mode) +
                      self.preferences['time_sensitivity'] * time_utility)
            
            mode_scores[mode] = utility
        
        # Choose mode with highest utility
        best_mode = max(mode_scores, key=lambda x: mode_scores[x])
        
        # Reset months counter if changing modes
        if best_mode != self.current_mode:
            self.months_with_current_mode = 0
        else:
            self.months_with_current_mode += 1
        
        # Check if agent wants to switch vehicles
        decision_reason = self._get_decision_reason(best_mode, gas_price)
        
        return best_mode, decision_reason
    
    def _get_emission_for_mode(self, mode: TransportMode, emissions: Emissions, daily_distance: float) -> float:
        """Calculate monthly emissions for a transport mode"""
        monthly_distance = daily_distance * 30
        
        if mode == TransportMode.WALKING:
            return emissions.walking_g_co2_per_km * monthly_distance
        elif mode == TransportMode.PUBLIC_TRANSIT:
            return emissions.public_transit_g_co2_per_km * monthly_distance
        elif mode == TransportMode.ICE_CAR:
            return emissions.ice_g_co2_per_km * monthly_distance
        elif mode == TransportMode.EV_CAR:
            return emissions.ev_g_co2_per_km * monthly_distance
        return 0
    
    def _get_convenience_score(self, mode: TransportMode) -> float:
        """Get convenience score for transport mode"""
        if mode == TransportMode.WALKING:
            return 0.3
        elif mode == TransportMode.PUBLIC_TRANSIT:
            return 0.6
        elif mode == TransportMode.ICE_CAR:
            return 0.9
        elif mode == TransportMode.EV_CAR:
            return 0.95
        return 0
    
    def _get_decision_reason(self, new_mode: TransportMode, gas_price: float) -> str:
        """Generate decision reason based on mode change"""
        if new_mode == self.current_mode:
            return f"continue with {self.current_mode.value.replace('_', ' ')}"
        
        if new_mode == TransportMode.EV_CAR and self.vehicle_owned == "ICE":
            return f"sell ICE car and purchase EV"
        elif new_mode == TransportMode.ICE_CAR and self.vehicle_owned == "EV":
            return f"sell EV and purchase ICE car"
        elif new_mode == TransportMode.PUBLIC_TRANSIT and self.vehicle_owned in ["ICE", "EV"]:
            return f"sell {self.vehicle_owned} car and switch to public transit"
        elif new_mode in [TransportMode.ICE_CAR, TransportMode.EV_CAR] and self.vehicle_owned == "None":
            vehicle_type = "EV" if new_mode == TransportMode.EV_CAR else "ICE"
            return f"purchase {vehicle_type} car"
        else:
            return f"switch to {new_mode.value.replace('_', ' ')}"

class CarbonPricingSimulation:
    def __init__(self):
        self.agents = []
        self.costs = TransportCosts()
        self.emissions = Emissions()
        self.month = 0
        self.history = {
            'mode_shares': [],
            'total_emissions': [],
            'ev_adoption_rate': [],
            'gas_prices': []
        }
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize 10 agents with different characteristics using ABC names"""
        names = ['Abigail', 'Bob', 'Carl', 'Diana', 'Eve', 'Frank', 'Grace', 'Henry', 'Ivy', 'Jack']
        
        # Income distribution: 2 low, 5 middle, 3 high (majority middle income)
        income_levels = [
            IncomeLevel.LOW, IncomeLevel.LOW,  # 2 low income
            IncomeLevel.MIDDLE, IncomeLevel.MIDDLE, IncomeLevel.MIDDLE,  # 5 middle income
            IncomeLevel.MIDDLE, IncomeLevel.MIDDLE,
            IncomeLevel.HIGH, IncomeLevel.HIGH, IncomeLevel.HIGH  # 3 high income
        ]
        
        # Initial modes with variety
        initial_modes = [
            TransportMode.PUBLIC_TRANSIT,  # Abigail - low income, public transit
            TransportMode.WALKING,         # Bob - low income, walking
            TransportMode.ICE_CAR,         # Carl - middle income, ICE car
            TransportMode.PUBLIC_TRANSIT,  # Diana - middle income, public transit
            TransportMode.ICE_CAR,         # Eve - middle income, ICE car
            TransportMode.EV_CAR,          # Frank - middle income, EV car
            TransportMode.ICE_CAR,         # Grace - middle income, ICE car
            TransportMode.EV_CAR,          # Henry - high income, EV car
            TransportMode.ICE_CAR,         # Ivy - high income, ICE car
            TransportMode.EV_CAR           # Jack - high income, EV car
        ]
        
        for i, name in enumerate(names):
            agent = Agent(name, income_levels[i], initial_modes[i])
            self.agents.append(agent)
    
    def run_month(self, carbon_pricing: CarbonPricing, daily_distance: float = 17.0):
        """Run simulation for one month"""
        self.month += 1
        gas_price = carbon_pricing.calculate_gas_price_per_gallon()
        
        print(f"\n=== MONTH {self.month} ===")
        print(f"Gas price: ${gas_price:.2f}/gallon (Oil: ${carbon_pricing.oil_price_per_barrel}/barrel, Carbon: ${carbon_pricing.carbon_price_per_gallon}/gallon)")
        
        # Agents make decisions
        total_emissions = 0
        mode_counts = {mode: 0 for mode in TransportMode}
        ev_count = 0
        
        for agent in self.agents:
            new_mode, reason = agent.make_decision(self.costs, carbon_pricing, self.emissions, daily_distance)
            
            # Update agent's mode and vehicle
            if new_mode in [TransportMode.ICE_CAR, TransportMode.EV_CAR]:
                agent.vehicle_owned = "EV" if new_mode == TransportMode.EV_CAR else "ICE"
            else:
                agent.vehicle_owned = "None"
            
            agent.current_mode = new_mode
            agent.decision_history.append((self.month, new_mode, reason))
            
            # Count modes and calculate emissions
            mode_counts[new_mode] += 1
            if new_mode == TransportMode.EV_CAR:
                ev_count += 1
            
            monthly_emissions = agent._get_emission_for_mode(new_mode, self.emissions, daily_distance)
            total_emissions += monthly_emissions
            
            # Calculate travel time for this agent
            travel_time = self.costs.get_travel_time(new_mode, daily_distance)
            monthly_cost = agent.calculate_mode_cost(new_mode, self.costs, gas_price, daily_distance)
            
            print(f"{agent.name} ({agent.income_level.value}, ${agent.monthly_income:.0f}/month, budget: ${agent.get_transit_budget():.0f}) decides to {reason}")
            print(f"  → Cost: ${monthly_cost:.0f}/month, Travel time: {travel_time:.1f}h/day, Emissions: {monthly_emissions/1000:.1f}kg CO2/month")
        
        # Calculate and display statistics
        total_agents = len(self.agents)
        mode_shares = {mode.value: (count/total_agents)*100 for mode, count in mode_counts.items()}
        ev_adoption_rate = (ev_count / total_agents) * 100
        
        print(f"\n--- STATISTICS ---")
        print(f"Mode shares:")
        for mode, share in mode_shares.items():
            print(f"  {mode.replace('_', ' ')}: {share:.1f}%")
        print(f"EV adoption rate: {ev_adoption_rate:.1f}%")
        print(f"Total monthly emissions: {total_emissions/1000:.1f} kg CO2")
        
        # Breakdown of emissions by mode
        print(f"Emissions breakdown:")
        for mode in TransportMode:
            mode_agents = [a for a in self.agents if a.current_mode == mode]
            mode_emissions = sum(a._get_emission_for_mode(mode, self.emissions, daily_distance) for a in mode_agents)
            if mode_emissions > 0:
                print(f"  {mode.value.replace('_', ' ')}: {mode_emissions/1000:.1f} kg CO2 ({mode_emissions/total_emissions*100:.1f}%)")
        
        # Store history
        self.history['mode_shares'].append(mode_shares)
        self.history['total_emissions'].append(total_emissions/1000)  # Convert to kg
        self.history['ev_adoption_rate'].append(ev_adoption_rate)
        self.history['gas_prices'].append(gas_price)
    
    def plot_trends(self):
        """Plot trends over time"""
        months = list(range(1, self.month + 1))
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Mode shares over time
        for mode in TransportMode:
            shares = [month_data[mode.value] for month_data in self.history['mode_shares']]
            ax1.plot(months, shares, label=mode.value.replace('_', ' '), marker='o')
        ax1.set_title('Mode Shares Over Time')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Share (%)')
        ax1.legend()
        ax1.grid(True)
        
        # Total emissions over time
        ax2.plot(months, self.history['total_emissions'], marker='o', color='red')
        ax2.set_title('Total Monthly Emissions')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Emissions (kg CO2)')
        ax2.grid(True)
        
        # EV adoption rate over time
        ax3.plot(months, self.history['ev_adoption_rate'], marker='o', color='green')
        ax3.set_title('EV Adoption Rate')
        ax3.set_xlabel('Month')
        ax3.set_ylabel('EV Adoption Rate (%)')
        ax3.grid(True)
        
        # Gas prices over time
        ax4.plot(months, self.history['gas_prices'], marker='o', color='orange')
        ax4.set_title('Gas Prices Over Time')
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Gas Price ($/gallon)')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()

def main():
    """Main simulation function"""
    print("=== CARBON PRICING SIMULATION ===")
    print("Small town transportation decision simulation with 10 agents")
    print("Agents make monthly decisions based on costs, preferences, and carbon pricing")
    
    # Initialize simulation
    sim = CarbonPricingSimulation()
    
    # Get user input for carbon pricing
    print("\nEnter simulation parameters:")
    oil_price = float(input("Oil price per barrel ($): ") or "80")
    carbon_price_per_gallon = float(input("Carbon price per gallon ($): ") or "0.2")
    
    carbon_pricing = CarbonPricing(
        oil_price_per_barrel=oil_price,
        carbon_price_per_gallon=carbon_price_per_gallon
    )
    
    print(f"\nStarting simulation with:")
    print(f"Oil price: ${oil_price}/barrel")
    print(f"Carbon price: ${carbon_price_per_gallon}/gallon")
    print(f"Gas price: ${carbon_pricing.calculate_gas_price_per_gallon():.2f}/gallon")
    
    # Run simulation
    month = 0
    while True:
        try:
            sim.run_month(carbon_pricing)
            month += 1
            
            if month % 3 == 0:  # Every 3 months, show trends
                print(f"\n{'='*50}")
                print(f"TREND SUMMARY (Month {month})")
                print(f"Total emissions trend: {sim.history['total_emissions'][-3:]}")
                print(f"EV adoption trend: {sim.history['ev_adoption_rate'][-3:]}")
                print(f"{'='*50}")
            
            # Ask user if they want to continue
            if month % 6 == 0:  # Every 6 months
                response = input(f"\nContinue simulation? (y/n/p for plot): ").lower()
                if response == 'n':
                    break
                elif response == 'p':
                    sim.plot_trends()
                    continue_response = input("Continue simulation? (y/n): ").lower()
                    if continue_response == 'n':
                        break
            
        except KeyboardInterrupt:
            print("\nSimulation paused. Options:")
            print("1. Continue (press Enter)")
            print("2. Show trends (type 'p')")
            print("3. Exit (type 'n')")
            
            response = input("Choice: ").lower()
            if response == 'n':
                break
            elif response == 'p':
                sim.plot_trends()
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SIMULATION SUMMARY")
    print(f"{'='*60}")
    print(f"Simulation ran for {month} months")
    print(f"Final gas price: ${sim.history['gas_prices'][-1]:.2f}/gallon")
    print(f"Final total emissions: {sim.history['total_emissions'][-1]:.1f} kg CO2/month")
    print(f"Final EV adoption rate: {sim.history['ev_adoption_rate'][-1]:.1f}%")
    
    # Show final mode distribution
    final_modes = {}
    for agent in sim.agents:
        mode = agent.current_mode.value
        final_modes[mode] = final_modes.get(mode, 0) + 1
    
    print(f"\nFinal mode distribution:")
    for mode, count in final_modes.items():
        print(f"  {mode.replace('_', ' ')}: {count} agents")
    
    # Plot final trends
    sim.plot_trends()

if __name__ == "__main__":
    main()
