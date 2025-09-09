from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from amazon_sales_flow.tools.custom_tool import (
    CleanDataTool, ProfileDataTool, BusinessInsightsTool
)
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class DataanalystCrew():
    """DataanalystCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]
    agents_config: str = "config/agents.yaml"
    tasks_config: str = "config/tasks.yaml"
    
    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def data_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config['DataEngineerAgent'], # type: ignore[index]
            tools=[CleanDataTool()],
            verbose=True
        )

    @agent
    def data_profiler(self) -> Agent:
        return Agent(
            config=self.agents_config['DataProfilerAgent'], # type: ignore[index]
            tools=[ProfileDataTool()],
            verbose=True
        )
    
    @agent
    def business_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['BusinessAnalystAgent'], # type: ignore[index]
            tools=[BusinessInsightsTool()],
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task

    # Tasks #
    @task
    def data_cleaning(self) -> Task:
        return Task(
            config=self.tasks_config['load_and_clean_data'], # type: ignore[index]
            output_file="clean_data.csv"
        )

    @task
    def data_profiling(self) -> Task:
        return Task(
            config=self.tasks_config['profile_data'], # type: ignore[index]
            output_file=["eda_report.html"],
            output_json="dataset_contract.json"
        )

    @task
    def business_insights(self) -> Task:
        return Task(
            config=self.tasks_config['generate_business_insights'], # type: ignore[index]
            output_file="insights.md"
        )
    


    @crew
    def crew(self) -> Crew:
        """Creates the DataanalystCrew crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            memory=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
