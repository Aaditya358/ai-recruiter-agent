import os
import json
from typing import List, Dict
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class EngagementAgent:
    def __init__(self):
        self.llm = OpenAI(
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="gpt-3.5-turbo"
        )
        
        self.outreach_prompt = PromptTemplate(
            input_variables=["candidate_name", "job_title", "company"],
            template="""
            Write a personalized, friendly recruitment outreach message to {candidate_name} for a {job_title} position at {company}.
            The message should be concise, professional but warm, and express genuine interest in their background.
            Keep it to 2-3 sentences.
            """
        )
        
        self.interest_assessment_prompt = PromptTemplate(
            input_variables=["candidate_response", "job_details"],
            template="""
            Based on the candidate's response below, assess their interest level in the job opportunity on a scale of 0-1.
            Consider enthusiasm, willingness to learn, and engagement level.
            
            Candidate Response: {candidate_response}
            Job Details: {job_details}
            
            Return ONLY a JSON object with:
            {{"interest_score": <0-1>, "indicators": [list of positive indicators], "concerns": [list of concerns]}}
            """
        )
        
        self.response_simulation_prompt = PromptTemplate(
            input_variables=["candidate_name", "candidate_bio", "message", "turn_number"],
            template="""
            You are {candidate_name}. {candidate_bio}
            
            You received this message from a recruiter:
            "{message}"
            
            This is turn {turn_number} of a conversation. Respond as this candidate would.
            Be realistic - show some interest but also ask clarifying questions or express some hesitation.
            Keep your response to 1-2 sentences.
            """
        )
        
        self.outreach_chain = LLMChain(llm=self.llm, prompt=self.outreach_prompt)
        self.response_chain = LLMChain(llm=self.llm, prompt=self.response_simulation_prompt)
        self.assessment_chain = LLMChain(llm=self.llm, prompt=self.interest_assessment_prompt)
    
    def engage_candidate(self, candidate: Dict, job_details: Dict, max_turns: int = 3) -> Dict:
        """Simulate conversational engagement with a candidate"""
        engagement_data = {
            "candidate_id": candidate.get('id', ''),
            "candidate_name": candidate.get('name', ''),
            "conversation": [],
            "interest_indicators": [],
            "interest_score": 0.0,
            "final_assessment": ""
        }
        
        try:
            # Step 1: Create initial outreach
            initial_message = self.outreach_chain.run(
                candidate_name=candidate.get('name', ''),
                job_title=job_details.get('title', ''),
                company=job_details.get('company', '')
            )
            
            engagement_data["conversation"].append({
                "turn": 1,
                "role": "recruiter",
                "message": initial_message.strip()
            })
            
            # Step 2: Simulate candidate responses and engagement
            for turn in range(2, max_turns + 1):
                # Get simulated candidate response
                response = self.response_chain.run(
                    candidate_name=candidate.get('name', ''),
                    candidate_bio=candidate.get('bio', ''),
                    message=initial_message.strip(),
                    turn_number=turn
                )
                
                engagement_data["conversation"].append({
                    "turn": turn,
                    "role": "candidate",
                    "message": response.strip()
                })
                
                # Assess interest from this response
                try:
                    interest_data = self._assess_interest(response, job_details)
                    engagement_data["interest_indicators"].append(interest_data)
                except:
                    engagement_data["interest_indicators"].append({"interest_score": 0.5, "indicators": [], "concerns": []})
            
            # Step 3: Calculate final interest score
            if engagement_data["interest_indicators"]:
                scores = [ind.get("interest_score", 0.5) for ind in engagement_data["interest_indicators"]]
                engagement_data["interest_score"] = sum(scores) / len(scores)
            else:
                engagement_data["interest_score"] = 0.5
            
            # Step 4: Generate final assessment
            engagement_data["final_assessment"] = self._generate_assessment(engagement_data)
            
        except Exception as e:
            print(f"Error in engagement: {e}")
            engagement_data["interest_score"] = 0.3
            engagement_data["final_assessment"] = f"Engagement incomplete due to error: {str(e)}"
        
        return engagement_data
    
    def _assess_interest(self, response: str, job_details: Dict) -> Dict:
        """Assess interest from response"""
        try:
            job_summary = f"{job_details.get('title', '')} at {job_details.get('company', '')}"
            result = self.assessment_chain.run(
                candidate_response=response,
                job_details=job_summary
            )
            
            # Parse JSON response
            result = result.strip()
            if result.startswith("```json"):
                result = result[7:]
            if result.startswith("```"):
                result = result[3:]
            if result.endswith("```"):
                result = result[:-3]
            result = result.strip()
            
            return json.loads(result)
        except Exception as e:
            print(f"Error assessing interest: {e}")
            return {"interest_score": 0.5, "indicators": [], "concerns": []}
    
    def _generate_assessment(self, engagement_data: Dict) -> str:
        """Generate assessment summary"""
        score = engagement_data["interest_score"]
        
        if score >= 0.8:
            return "High interest - Candidate is very engaged and enthusiastic"
        elif score >= 0.6:
            return "Moderate-High interest - Candidate shows genuine curiosity"
        elif score >= 0.4:
            return "Moderate interest - Candidate is considering the opportunity"
        elif score >= 0.2:
            return "Low interest - Candidate seems hesitant"
        else:
            return "Very Low interest - Not a good cultural fit"