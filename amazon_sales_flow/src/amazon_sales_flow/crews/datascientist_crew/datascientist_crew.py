from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from amazon_sales_flow.tools.custom_tool import (
    FeatureEngineeringTool, ModelTrainingTool, ModelEvaluationTool, ModelCardTool
)
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class DatascientistCrew():
    """DatascientistCrew crew"""

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
    def feature_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config['FeatureEngineerAgent'],
            tools=[FeatureEngineeringTool()],
            verbose=true
        )

    @agent
    def model_trainer(self) -> Agent:
        return Agent(
            config=self.agents_config['ModelTrainerAgent'],
            tools=[ModelTrainingTool()],
            verbose=True
        )

    @agent
    def model_evaluator(self) -> Agent:
        return Agent(
            config=self.agents_config['ModelEvaluatorAgent'],
            tools=[ModelEvaluationTool()],
            verbose=True
        )

    @agent
    def model_card_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['ModelCardAgent'],
            tools=[ModelCardTool()],
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    
    @task
    def feature_engineering_task(self) -> Task:
        return Task(
            config=self.tasks_config['feature_engineering_task'],
            output_file='features.csv'
        )

    @task
    def model_training_task(self) -> Task:
        return Task(
            config=self.tasks_config['model_training_task'],
            output_file='trained_model.pkl'
        )

    @task
    def evaluation_task(self) -> Task:
        return Task(
            config=self.tasks_config['evaluation_task'],
            output_file='evaluation_report.md'
        )

    @task
    def model_card_task(self) -> Task:
        return Task(
            config=self.tasks_config['model_card_task'],
            output_file='model_card.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the DatascientistCrew crew"""
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
