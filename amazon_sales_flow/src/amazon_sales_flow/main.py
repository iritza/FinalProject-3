#!/usr/bin/env python
from pydantic import BaseModel

from crewai.flow import Flow, listen, start

from amazon_sales_flow.crews.dataanalyst_crew.dataanalyst_crew import DataanalystCrew
from amazon_sales_flow.crews.datascientist_crew.datascientist_crew import DatascientistCrew


class AmazonSalesState(BaseModel):
    data_cleaned: bool = False
    data_profiled: bool = False
    business_insights_generated: bool = False
    features_engineered: bool = False
    model_trained: bool = False
    model_evaluated: bool = False
    model_card_generated: bool = False


class AmazonSalesFlow(Flow[AmazonSalesState]):

    @start()
    def data_analysis_phase(self):
        print("Starting Amazon Sales Data Analysis Phase...")
        result = DataanalystCrew().crew().kickoff()
        print("Data Analysis Phase completed:", result.raw)
        self.state.data_cleaned = True
        self.state.data_profiled = True
        self.state.business_insights_generated = True

    # @listen(data_analysis_phase)
    # def data_science_phase(self):
    #     print("Starting Amazon Sales Data Science Phase...")
    #     result = DatascientistCrew().crew().kickoff()
    #     print("Data Science Phase completed:", result.raw)
    #     self.state.features_engineered = True
    #     self.state.model_trained = True
    #     self.state.model_evaluated = True
    #     self.state.model_card_generated = True

    # @listen(data_science_phase)
    # def finalize_analysis(self):
    #     print("Amazon Sales Analysis Complete!")
    #     print("Generated files:")
    #     print("- clean_data.csv")
    #     print("- eda_report.html")
    #     print("- dataset_contract.json")
    #     print("- insights.md")
    #     print("- features.csv")
    #     print("- trained_model.pkl")
    #     print("- evaluation_report.md")
    #     print("- model_card.md")


def kickoff():
    sales_flow = AmazonSalesFlow()
    sales_flow.kickoff()


def plot():
    sales_flow = AmazonSalesFlow()
    sales_flow.plot()


if __name__ == "__main__":
    kickoff()
