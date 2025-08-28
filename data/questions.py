import json
import random
import streamlit as st
from typing import List, Dict, Optional
import time
from transformers import pipeline
import torch

class AdaptiveMCQGenerator:
    def __init__(self):
        """Initialize the adaptive MCQ generator"""
        self.device = "cpu"
        
        # Initialize text generation pipeline for custom questions
        try:
            self.generator = pipeline(
                "text-generation",
                model="gpt2",
                device=0 if torch.cuda.is_available() else -1,
                pad_token_id=50256
            )
        except Exception as e:
            print(f"Warning: Could not initialize text generator: {e}")
            self.generator = None
        
        # Cache for generated questions to avoid duplicates
        self.question_cache = {}
        
        # Enhanced question pools with more variety
        self.question_pools = {
            "web": {
                1: [  # Easy
                    {
                        'text': "Which language is the core of web page structure?",
                        'options': ["HTML", "Python", "Java", "C++"],
                        'correct_answer': "HTML",
                        'explanation': "HTML (HyperText Markup Language) defines the structure and content of web pages."
                    },
                    {
                        'text': "What does CSS stand for?",
                        'options': ["Computer Style Sheets", "Creative Style Sheets", 
                                  "Cascading Style Sheets", "Colorful Style Sheets"],
                        'correct_answer': "Cascading Style Sheets",
                        'explanation': "CSS controls the visual styling and layout of web pages."
                    },
                    {
                        'text': "Which HTML tag is used for paragraphs?",
                        'options': ["<p>", "<para>", "<paragraph>", "<pg>"],
                        'correct_answer': "<p>",
                        'explanation': "The <p> tag defines a paragraph in HTML."
                    },
                    {
                        'text': "What is the correct way to comment in HTML?",
                        'options': ["<!-- comment -->", "// comment", "/* comment */", "# comment"],
                        'correct_answer': "<!-- comment -->",
                        'explanation': "HTML comments use <!-- --> syntax."
                    },
                    {
                        'text': "Which tag creates a hyperlink?",
                        'options': ["<link>", "<a>", "<href>", "<url>"],
                        'correct_answer': "<a>",
                        'explanation': "The <a> (anchor) tag creates hyperlinks in HTML."
                    },
                    {
                        'text': "What file extension is used for CSS files?",
                        'options': [".css", ".style", ".htm", ".web"],
                        'correct_answer': ".css",
                        'explanation': "CSS files use the .css extension."
                    }
                ],
                2: [  # Medium
                    {
                        'text': "Which React hook is used for side effects?",
                        'options': ["useState", "useEffect", "useContext", "useReducer"],
                        'correct_answer': "useEffect",
                        'explanation': "useEffect handles side effects like API calls and subscriptions."
                    },
                    {
                        'text': "What does API stand for?",
                        'options': ["Application Programming Interface", "Automated Programming Interface",
                                  "Advanced Programming Instruction", "Application Process Integration"],
                        'correct_answer': "Application Programming Interface",
                        'explanation': "APIs allow different software applications to communicate."
                    },
                    {
                        'text': "Which method adds a new element to an array?",
                        'options': ["array.add()", "array.push()", "array.insert()", "array.append()"],
                        'correct_answer': "array.push()",
                        'explanation': "push() adds elements to the end of an array in JavaScript."
                    },
                    {
                        'text': "What is the box model in CSS?",
                        'options': ["A way to package CSS files", "A layout concept with margin, border, padding",
                                    "A 3D modeling technique", "A JavaScript framework"],
                        'correct_answer': "A layout concept with margin, border, padding",
                        'explanation': "The CSS box model describes how elements are rendered with content, padding, border, and margin."
                    },
                    {
                        'text': "What is the difference between GET and POST requests?",
                        'options': ["GET retrieves data, POST sends data", "GET is faster than POST",
                                  "POST is more secure than GET", "No difference"],
                        'correct_answer': "GET retrieves data, POST sends data",
                        'explanation': "GET requests retrieve data while POST requests send data to the server."
                    }
                ],
                3: [  # Hard
                    {
                        'text': "Which HTTP status code indicates 'Not Found'?",
                        'options': ["200", "301", "404", "500"],
                        'correct_answer': "404",
                        'explanation': "404 indicates that the requested resource was not found on the server."
                    },
                    {
                        'text': "What is the virtual DOM in React?",
                        'options': ["A lightweight version of the real DOM", "A 3D visualization tool",
                                  "A security feature", "A testing framework"],
                        'correct_answer': "A lightweight version of the real DOM",
                        'explanation': "Virtual DOM is a JavaScript representation of the real DOM for better performance."
                    },
                    {
                        'text': "What is CORS in web development?",
                        'options': ["Cross-Origin Resource Sharing", "Computer Object Recognition System",
                                  "Cascading Object Rendering Style", "Component Object Resource System"],
                        'correct_answer': "Cross-Origin Resource Sharing",
                        'explanation': "CORS is a security feature that controls how web pages access resources from other domains."
                    },
                    {
                        'text': "What does SSR stand for in Next.js?",
                        'options': ["Server-Side Rendering", "Static Site Rendering",
                                  "Secure Socket Rendering", "System Style Rules"],
                        'correct_answer': "Server-Side Rendering",
                        'explanation': "SSR renders pages on the server before sending them to the client."
                    },
                    {
                        'text': "What is the purpose of Webpack?",
                        'options': ["Module bundling and optimization", "Database management",
                                  "User authentication", "Email services"],
                        'correct_answer': "Module bundling and optimization",
                        'explanation': "Webpack bundles JavaScript modules and optimizes assets for production."
                    }
                ]
            },
            
            "ai": {
                1: [  # Easy
                    {
                        'text': "What is the most common programming language for AI?",
                        'options': ["Python", "C#", "Ruby", "Go"],
                        'correct_answer': "Python",
                        'explanation': "Python is widely used in AI due to its simplicity and extensive libraries."
                    },
                    {
                        'text': "What does AI stand for?",
                        'options': ["Automated Intelligence", "Artificial Intelligence",
                                  "Advanced Interface", "Algorithmic Inference"],
                        'correct_answer': "Artificial Intelligence",
                        'explanation': "AI refers to machines that can perform tasks typically requiring human intelligence."
                    },
                    {
                        'text': "Which library is commonly used for machine learning in Python?",
                        'options': ["scikit-learn", "Django", "Flask", "Beautiful Soup"],
                        'correct_answer': "scikit-learn",
                        'explanation': "scikit-learn is a popular machine learning library in Python."
                    },
                    {
                        'text': "What is machine learning?",
                        'options': ["Teaching machines to learn from data", "Programming robots",
                                  "Creating websites", "Database management"],
                        'correct_answer': "Teaching machines to learn from data",
                        'explanation': "Machine learning enables computers to learn and improve from experience without explicit programming."
                    }
                ],
                2: [  # Medium
                    {
                        'text': "Which algorithm is best for classification tasks?",
                        'options': ["Linear Regression", "K-Means", "Decision Tree", "DBSCAN"],
                        'correct_answer': "Decision Tree",
                        'explanation': "Decision Trees are excellent for classification problems due to their interpretability."
                    },
                    {
                        'text': "What is supervised learning?",
                        'options': ["Learning with labeled data", "Learning without guidance",
                                  "Learning from rewards", "Learning from unstructured data"],
                        'correct_answer': "Learning with labeled data",
                        'explanation': "Supervised learning uses labeled datasets to train models to make predictions."
                    },
                    {
                        'text': "What is the difference between training and testing data?",
                        'options': ["Training data teaches the model, testing data evaluates it", 
                                  "No difference", "Testing data is larger", "Training data is more accurate"],
                        'correct_answer': "Training data teaches the model, testing data evaluates it",
                        'explanation': "Training data is used to build the model, testing data measures its performance."
                    },
                    {
                        'text': "What is overfitting in machine learning?",
                        'options': ["Model performs well on training but poorly on new data", "Model is too simple",
                                  "Model trains too slowly", "Model uses too much memory"],
                        'correct_answer': "Model performs well on training but poorly on new data",
                        'explanation': "Overfitting occurs when a model memorizes training data instead of learning patterns."
                    }
                ],
                3: [  # Hard
                    {
                        'text': "What does CNN stand for in deep learning?",
                        'options': ["Common Neural Network", "Convolutional Neural Network",
                                  "Complex Neural Node", "Centralized Neural Network"],
                        'correct_answer': "Convolutional Neural Network",
                        'explanation': "CNNs are specialized for processing grid-like data such as images."
                    },
                    {
                        'text': "What is the purpose of backpropagation?",
                        'options': ["To adjust neural network weights", "To collect data",
                                  "To visualize results", "To preprocess input"],
                        'correct_answer': "To adjust neural network weights",
                        'explanation': "Backpropagation calculates gradients to update network weights during training."
                    },
                    {
                        'text': "What is transfer learning?",
                        'options': ["Reusing a pre-trained model for new tasks", "Moving data between servers",
                                  "Changing programming languages", "Converting code to binary"],
                        'correct_answer': "Reusing a pre-trained model for new tasks",
                        'explanation': "Transfer learning leverages knowledge from pre-trained models for related tasks."
                    },
                    {
                        'text': "What is gradient descent?",
                        'options': ["An optimization algorithm", "A data preprocessing technique",
                                  "A neural network architecture", "A programming language"],
                        'correct_answer': "An optimization algorithm",
                        'explanation': "Gradient descent is used to minimize the loss function by iteratively adjusting parameters."
                    }
                ]
            },
            
            "cyber": {
                1: [  # Easy
                    {
                        'text': "What is the most common type of cyber attack?",
                        'options': ["Phishing", "DDoS", "MITM", "Zero-day"],
                        'correct_answer': "Phishing",
                        'explanation': "Phishing attacks trick users into revealing sensitive information through fake communications."
                    },
                    {
                        'text': "What is a firewall used for?",
                        'options': ["Network security", "Data storage", "Website design", "Programming"],
                        'correct_answer': "Network security",
                        'explanation': "Firewalls monitor and control incoming and outgoing network traffic."
                    },
                    {
                        'text': "What does HTTPS stand for?",
                        'options': ["HyperText Transfer Protocol Secure", "Hyper Transfer Text Protocol",
                                  "High-Tech Transfer Protocol", "HyperText Transfer Protocol Standard"],
                        'correct_answer': "HyperText Transfer Protocol Secure",
                        'explanation': "HTTPS is the secure version of HTTP that encrypts data transmission."
                    },
                    {
                        'text': "What is malware?",
                        'options': ["Malicious software", "Mail software", "Main software", "Manual software"],
                        'correct_answer': "Malicious software",
                        'explanation': "Malware is software designed to damage or gain unauthorized access to systems."
                    }
                ],
                2: [  # Medium
                    {
                        'text': "What does VPN stand for?",
                        'options': ["Virtual Private Network", "Verified Personal Node",
                                  "Virtual Public Network", "Verified Protocol Network"],
                        'correct_answer': "Virtual Private Network",
                        'explanation': "VPNs create secure connections over public networks."
                    },
                    {
                        'text': "What is two-factor authentication?",
                        'options': ["An extra security layer requiring two forms of verification", "A type of virus",
                                  "A network protocol", "A programming language"],
                        'correct_answer': "An extra security layer requiring two forms of verification",
                        'explanation': "2FA adds security by requiring something you know and something you have."
                    },
                    {
                        'text': "What is encryption?",
                        'options': ["Converting data into a coded format", "Deleting files",
                                  "Creating backups", "Formatting disks"],
                        'correct_answer': "Converting data into a coded format",
                        'explanation': "Encryption transforms readable data into an unreadable format for security."
                    },
                    {
                        'text': "What is social engineering?",
                        'options': ["Manipulating people to divulge information", "Building social networks",
                                  "Creating user interfaces", "Developing databases"],
                        'correct_answer': "Manipulating people to divulge information",
                        'explanation': "Social engineering exploits human psychology rather than technical vulnerabilities."
                    }
                ],
                3: [  # Hard
                    {
                        'text': "Which encryption algorithm is asymmetric?",
                        'options': ["AES", "RSA", "DES", "3DES"],
                        'correct_answer': "RSA",
                        'explanation': "RSA uses different keys for encryption and decryption (public/private key pairs)."
                    },
                    {
                        'text': "What is a zero-day vulnerability?",
                        'options': ["A security flaw unknown to vendors", "A type of firewall",
                                  "An encryption method", "A network protocol"],
                        'correct_answer': "A security flaw unknown to vendors",
                        'explanation': "Zero-day vulnerabilities are unknown to security vendors and have no available patches."
                    },
                    {
                        'text': "What is penetration testing?",
                        'options': ["Authorized security testing", "Data encryption",
                                  "Virus scanning", "Network monitoring"],
                        'correct_answer': "Authorized security testing",
                        'explanation': "Pen testing simulates attacks to identify security weaknesses."
                    },
                    {
                        'text': "What is SQL injection?",
                        'options': ["Malicious database queries", "Database optimization",
                                  "Data backup method", "Network configuration"],
                        'correct_answer': "Malicious database queries",
                        'explanation': "SQL injection attacks insert malicious code into database queries."
                    }
                ]
            },
            
            "data": {
                1: [  # Easy
                    {
                        'text': "What is the primary tool for data analysis in Python?",
                        'options': ["Pandas", "Django", "TensorFlow", "Flask"],
                        'correct_answer': "Pandas",
                        'explanation': "Pandas provides data structures and analysis tools for Python."
                    },
                    {
                        'text': "What is a DataFrame?",
                        'options': ["2D data structure", "A chart type", "A database", "A file format"],
                        'correct_answer': "2D data structure",
                        'explanation': "DataFrame is a 2D labeled data structure similar to a spreadsheet."
                    },
                    {
                        'text': "What does CSV stand for?",
                        'options': ["Comma Separated Values", "Computer System Variables",
                                  "Columnar Storage Version", "Coded Secure Variables"],
                        'correct_answer': "Comma Separated Values",
                        'explanation': "CSV files store tabular data with comma-separated values."
                    },
                    {
                        'text': "What is data science?",
                        'options': ["Extracting insights from data", "Computer programming",
                                  "Web development", "Network administration"],
                        'correct_answer': "Extracting insights from data",
                        'explanation': "Data science combines statistics, programming, and domain knowledge to analyze data."
                    }
                ],
                2: [  # Medium
                    {
                        'text': "Which type of variable is 'age'?",
                        'options': ["Categorical", "Numerical", "Ordinal", "Binary"],
                        'correct_answer': "Numerical",
                        'explanation': "Age is a numerical (quantitative) variable that can be measured."
                    },
                    {
                        'text': "What is data cleaning?",
                        'options': ["Fixing errors and inconsistencies in data", "Deleting old data",
                                  "Encrypting data", "Visualizing data"],
                        'correct_answer': "Fixing errors and inconsistencies in data",
                        'explanation': "Data cleaning involves identifying and correcting errors in datasets."
                    },
                    {
                        'text': "What is the purpose of Matplotlib?",
                        'options': ["Data visualization", "Web development",
                                  "Machine learning", "Database management"],
                        'correct_answer': "Data visualization",
                        'explanation': "Matplotlib is a Python library for creating static, animated, and interactive visualizations."
                    },
                    {
                        'text': "What is the difference between mean and median?",
                        'options': ["Mean is average, median is middle value", "No difference",
                                  "Median is always larger", "Mean is more accurate"],
                        'correct_answer': "Mean is average, median is middle value",
                        'explanation': "Mean is the arithmetic average; median is the middle value when data is sorted."
                    }
                ],
                3: [  # Hard
                    {
                        'text': "What does EDA stand for in data science?",
                        'options': ["Electronic Data Analysis", "Exploratory Data Analysis",
                                  "Estimated Data Assessment", "Extended Data Algorithm"],
                        'correct_answer': "Exploratory Data Analysis",
                        'explanation': "EDA is the process of analyzing datasets to summarize their main characteristics."
                    },
                    {
                        'text': "What is feature engineering?",
                        'options': ["Creating better input features for models", "Building UI features",
                                  "Designing databases", "Writing documentation"],
                        'correct_answer': "Creating better input features for models",
                        'explanation': "Feature engineering involves creating new features from existing data to improve model performance."
                    },
                    {
                        'text': "What is cross-validation?",
                        'options': ["A technique to assess model performance", "Data cleaning method",
                                  "Visualization technique", "Database optimization"],
                        'correct_answer': "A technique to assess model performance",
                        'explanation': "Cross-validation evaluates model performance by training on different subsets of data."
                    },
                    {
                        'text': "What is the curse of dimensionality?",
                        'options': ["Problems arising with high-dimensional data", "Database corruption",
                                  "Network latency issues", "Memory overflow"],
                        'correct_answer': "Problems arising with high-dimensional data",
                        'explanation': "High-dimensional data becomes sparse, making analysis and modeling difficult."
                    }
                ]
            },
            
            "mobile": {
                1: [  # Easy
                    {
                        'text': "Which language is used for native Android development?",
                        'options': ["Swift", "Kotlin", "C#", "JavaScript"],
                        'correct_answer': "Kotlin",
                        'explanation': "Kotlin is Google's preferred language for Android development."
                    },
                    {
                        'text': "What is Flutter used for?",
                        'options': ["Cross-platform mobile apps", "Web development",
                                  "Game development", "Data analysis"],
                        'correct_answer': "Cross-platform mobile apps",
                        'explanation': "Flutter allows building apps for multiple platforms from a single codebase."
                    },
                    {
                        'text': "Which language is used for iOS development?",
                        'options': ["Java", "Kotlin", "Swift", "Dart"],
                        'correct_answer': "Swift",
                        'explanation': "Swift is Apple's programming language for iOS app development."
                    },
                    {
                        'text': "What is an IDE?",
                        'options': ["Integrated Development Environment", "Internet Data Exchange",
                                  "Interactive Design Engine", "Internal Database Engine"],
                        'correct_answer': "Integrated Development Environment",
                        'explanation': "IDEs provide comprehensive tools for software development."
                    }
                ],
                2: [  # Medium
                    {
                        'text': "What is React Native used for?",
                        'options': ["Cross-platform mobile development", "Web development only",
                                  "Game development", "Data analysis"],
                        'correct_answer': "Cross-platform mobile development",
                        'explanation': "React Native uses JavaScript to build native mobile apps for multiple platforms."
                    },
                    {
                        'text': "What is an APK file?",
                        'options': ["Android application package", "Apple program kit",
                                  "Application programming key", "Automated process kernel"],
                        'correct_answer': "Android application package",
                        'explanation': "APK files are the package format for Android applications."
                    },
                    {
                        'text': "What is Jetpack Compose?",
                        'options': ["Modern UI toolkit for Android", "iOS development framework",
                                  "Cross-platform solution", "Backend technology"],
                        'correct_answer': "Modern UI toolkit for Android",
                        'explanation': "Jetpack Compose is Android's modern declarative UI toolkit."
                    },
                    {
                        'text': "What is the difference between native and hybrid apps?",
                        'options': ["Native uses platform-specific code, hybrid uses web technologies",
                                  "No difference", "Hybrid apps are faster", "Native apps are web-based"],
                        'correct_answer': "Native uses platform-specific code, hybrid uses web technologies",
                        'explanation': "Native apps are built for specific platforms; hybrid apps use web technologies wrapped in native containers."
                    }
                ],
                3: [  # Hard
                    {
                        'text': "Which architecture pattern is recommended for Android apps?",
                        'options': ["MVVM", "Singleton", "Factory", "Observer"],
                        'correct_answer': "MVVM",
                        'explanation': "MVVM (Model-View-ViewModel) separates UI logic from business logic in Android apps."
                    },
                    {
                        'text': "What is Firebase commonly used for in mobile apps?",
                        'options': ["Backend services and cloud infrastructure", "UI design",
                                  "Game engines", "Local storage only"],
                        'correct_answer': "Backend services and cloud infrastructure",
                        'explanation': "Firebase provides authentication, databases, hosting, and other backend services."
                    },
                    {
                        'text': "What is the purpose of Gradle in Android development?",
                        'options': ["Build automation and dependency management", "UI design",
                                  "Database management", "Networking only"],
                        'correct_answer': "Build automation and dependency management",
                        'explanation': "Gradle automates the build process and manages project dependencies."
                    },
                    {
                        'text': "What is dependency injection in mobile development?",
                        'options': ["A design pattern for managing dependencies", "A security feature",
                                  "A testing framework", "A UI component"],
                        'correct_answer': "A design pattern for managing dependencies",
                        'explanation': "Dependency injection provides dependencies to objects rather than having them create dependencies themselves."
                    }
                ]
            },
            
            "devops": {
                1: [  # Easy
                    {
                        'text': "What is Docker used for?",
                        'options': ["Containerization", "Database management", "Web design", "Mobile development"],
                        'correct_answer': "Containerization",
                        'explanation': "Docker packages applications and their dependencies into lightweight containers."
                    },
                    {
                        'text': "What does CI/CD stand for?",
                        'options': ["Continuous Integration/Continuous Deployment", "Computer Integration/Computer Deployment",
                                  "Code Integration/Code Deployment", "Continuous Internet/Continuous Data"],
                        'correct_answer': "Continuous Integration/Continuous Deployment",
                        'explanation': "CI/CD automates the process of integrating code changes and deploying applications."
                    },
                    {
                        'text': "What is version control?",
                        'options': ["Tracking changes in code", "Controlling software versions",
                                  "Managing databases", "Network monitoring"],
                        'correct_answer': "Tracking changes in code",
                        'explanation': "Version control systems track and manage changes to source code over time."
                    }
                ],
                2: [  # Medium
                    {
                        'text': "What is Infrastructure as Code (IaC)?",
                        'options': ["Managing infrastructure through code", "Writing application code",
                                  "Database programming", "Web development"],
                        'correct_answer': "Managing infrastructure through code",
                        'explanation': "IaC uses code to provision and manage infrastructure resources automatically."
                    },
                    {
                        'text': "Which tool is commonly used for configuration management?",
                        'options': ["Ansible", "Photoshop", "Word", "Excel"],
                        'correct_answer': "Ansible",
                        'explanation': "Ansible automates configuration management and application deployment."
                    }
                ],
                3: [  # Hard
                    {
                        'text': "What is Kubernetes used for?",
                        'options': ["Container orchestration", "Database management",
                                  "Web design", "Mobile development"],
                        'correct_answer': "Container orchestration",
                        'explanation': "Kubernetes automates deployment, scaling, and management of containerized applications."
                    },
                    {
                        'text': "What is the difference between horizontal and vertical scaling?",
                        'options': ["Horizontal adds more machines, vertical adds more power",
                                  "No difference", "Vertical is always better", "Horizontal is cheaper"],
                        'correct_answer': "Horizontal adds more machines, vertical adds more power",
                        'explanation': "Horizontal scaling adds more servers; vertical scaling increases the power of existing servers."
                    }
                ]
            }
        }

    def get_available_tracks(self) -> List[str]:
        """Get list of available technology tracks"""
        return list(self.question_pools.keys())

    def get_question(self, track: str, level: int, exclude_used: List[str] = None) -> Optional[Dict]:
        """
        Get a question from the specified track and level
        
        Args:
            track: Technology track (web, ai, cyber, data, mobile, devops)
            level: Difficulty level (1=easy, 2=medium, 3=hard)
            exclude_used: List of question texts to exclude (to avoid duplicates)
        
        Returns:
            Question dictionary or None if no questions available
        """
        if track not in self.question_pools:
            return None
            
        if level not in self.question_pools[track]:
            return None
        
        available_questions = self.question_pools[track][level].copy()
        
        # Exclude already used questions
        if exclude_used:
            available_questions = [
                q for q in available_questions 
                if q['text'] not in exclude_used
            ]
        
        if not available_questions:
            return None
            
        return random.choice(available_questions)

    def generate_custom_question(self, track: str, level: int, topic: str = None) -> Dict:
        """
        Generate a custom question using AI or fallback to templates
        
        Args:
            track: Technology track
            level: Difficulty level
            topic: Specific topic (optional)
        
        Returns:
            Generated question dictionary
        """
        difficulty_names = {1: "easy", 2: "medium", 3: "hard"}
        difficulty = difficulty_names.get(level, "medium")
        
        # Try to use AI generation if available
        if self.generator and topic:
            try:
                prompt = f"Create a {difficulty} {track} question about {topic}:\nQ:"
                response = self.generator(
                    prompt,
                    max_length=150,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=50256
                )
                
                generated_text = response[0]['generated_text'].replace(prompt, "").strip()
                question_text = generated_text.split('\n')[0] if generated_text else f"What is {topic} in {track}?"
                
                # Generate reasonable options
                options = self._generate_options(track, topic, difficulty)
                
                return {
                    'text': question_text,
                    'options': options,
                    'correct_answer': options[0],  # First option is correct by design
                    'explanation': f"This question tests {difficulty} level knowledge of {topic} in {track}.",
                    'generated': True,
                    'topic': topic
                }
                
            except Exception as e:
                print(f"AI generation failed: {e}")
        
        # Fallback to template-based generation
        return self._generate_template_question(track, level, topic)

    def _generate_options(self, track: str, topic: str, difficulty: str) -> List[str]:
        """Generate plausible options for a given track and topic"""
        option_templates = {
            "web": {
                "easy": [f"Correct answer about {topic}", "Incorrect web concept", "Unrelated technology", "Common misconception"],
                "medium": [f"Advanced {topic} concept", "Related but incorrect", "Different framework", "Outdated approach"],
                "hard": [f"Complex {topic} implementation", "Advanced alternative", "Edge case scenario", "Theoretical concept"]
            },
            "ai": {
                "easy": [f"Basic {topic} principle", "Unrelated AI concept", "Different algorithm", "Common mistake"],
                "medium": [f"Practical {topic} application", "Related technique", "Alternative approach", "Simplified version"],
                "hard": [f"Advanced {topic} theory", "Complex implementation", "Research-level concept", "Cutting-edge method"]
            },
            "cyber": {
                "easy": [f"Basic {topic} concept", "Different security tool", "Unrelated technology", "Common vulnerability"],
                "medium": [f"Practical {topic} use", "Related security measure", "Alternative protocol", "Security framework"],
                "hard": [f"Advanced {topic} technique", "Complex attack vector", "Enterprise security solution", "Theoretical vulnerability"]
            },
            "data": {
                "easy": [f"Basic {topic} definition", "Different data tool", "Unrelated concept", "Common data type"],
                "medium": [f"Practical {topic} application", "Related analysis method", "Alternative technique", "Statistical concept"],
                "hard": [f"Advanced {topic} theory", "Complex algorithm", "Research methodology", "Machine learning approach"]
            },
            "mobile": {
                "easy": [f"Basic {topic} concept", "Different platform", "Unrelated technology", "Common framework"],
                "medium": [f"Practical {topic} implementation", "Related pattern", "Alternative approach", "Development tool"],
                "hard": [f"Advanced {topic} architecture", "Complex design pattern", "Performance optimization", "Enterprise solution"]
            },
            "devops": {
                "easy": [f"Basic {topic} concept", "Different tool", "Unrelated technology", "Common practice"],
                "medium": [f"Practical {topic} use", "Related methodology", "Alternative tool", "Automation technique"],
                "hard": [f"Advanced {topic} strategy", "Complex architecture", "Enterprise implementation", "Scalability solution"]
            }
        }
        
        templates = option_templates.get(track, {}).get(difficulty, [f"Option about {topic}", "Alternative concept", "Different approach", "Related idea"])
        return templates[:4]  # Return first 4 options

    def _generate_template_question(self, track: str, level: int, topic: str = None) -> Dict:
        """Generate a question using templates when AI generation fails"""
        difficulty_names = {1: "easy", 2: "medium", 3: "hard"}
        difficulty = difficulty_names.get(level, "medium")
        
        topic = topic or f"{track} concepts"
        
        question_templates = {
            1: f"What is a fundamental concept in {topic}?",
            2: f"How would you implement {topic} in a real project?", 
            3: f"What are the advanced considerations when working with {topic}?"
        }
        
        question_text = question_templates.get(level, f"What do you know about {topic}?")
        options = self._generate_options(track, topic, difficulty)
        
        return {
            'text': question_text,
            'options': options,
            'correct_answer': options[0],
            'explanation': f"This is a {difficulty} level question about {topic} in {track}.",
            'generated': True,
            'topic': topic
        }

    def get_adaptive_question_set(self, track: str, student_ability: float, num_questions: int = 10) -> List[Dict]:
        """
        Generate an adaptive question set based on student ability
        
        Args:
            track: Technology track
            student_ability: Current student ability (0.0 to 1.0)
            num_questions: Number of questions to generate
        
        Returns:
            List of questions adapted to student level
        """
        questions = []
        used_questions = set()
        
        # Determine level distribution based on student ability
        if student_ability < 0.3:  # Beginner
            level_distribution = [0.6, 0.3, 0.1]  # 60% easy, 30% medium, 10% hard
        elif student_ability < 0.7:  # Intermediate
            level_distribution = [0.2, 0.6, 0.2]  # 20% easy, 60% medium, 20% hard
        else:  # Advanced
            level_distribution = [0.1, 0.3, 0.6]  # 10% easy, 30% medium, 60% hard
        
        # Calculate number of questions per level
        easy_count = int(num_questions * level_distribution[0])
        medium_count = int(num_questions * level_distribution[1])
        hard_count = num_questions - easy_count - medium_count
        
        # Generate questions for each level
        for level, count in [(1, easy_count), (2, medium_count), (3, hard_count)]:
            for _ in range(count):
                question = self.get_question(track, level, list(used_questions))
                if question:
                    questions.append(question)
                    used_questions.add(question['text'])
                elif len(questions) < num_questions:
                    # Generate custom question if pool is exhausted
                    custom_q = self.generate_custom_question(track, level)
                    questions.append(custom_q)
                    used_questions.add(custom_q['text'])
        
        # Shuffle questions to avoid predictable patterns
        random.shuffle(questions)
        return questions

    def update_question_pool(self, track: str, level: int, new_questions: List[Dict]):
        """Add new questions to the pool dynamically"""
        if track not in self.question_pools:
            self.question_pools[track] = {}
        if level not in self.question_pools[track]:
            self.question_pools[track][level] = []
        
        self.question_pools[track][level].extend(new_questions)

    def get_track_statistics(self, track: str) -> Dict:
        """Get statistics about questions in a track"""
        if track not in self.question_pools:
            return {"error": "Track not found"}
        
        stats = {"track": track, "levels": {}}
        
        for level, questions in self.question_pools[track].items():
            stats["levels"][level] = {
                "count": len(questions),
                "difficulty": {1: "Easy", 2: "Medium", 3: "Hard"}.get(level, "Unknown")
            }
        
        stats["total_questions"] = sum(len(questions) for questions in self.question_pools[track].values())
        return stats

    def export_questions(self, track: str = None, filename: str = None) -> str:
        """Export questions to JSON file"""
        if not filename:
            timestamp = int(time.time())
            filename = f"questions_{track or 'all'}_{timestamp}.json"
        
        data_to_export = self.question_pools[track] if track else self.question_pools
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data_to_export, f, indent=2, ensure_ascii=False)
        
        return filename

    def load_questions_from_file(self, filename: str):
        """Load additional questions from JSON file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Merge with existing questions
            for track, levels in data.items():
                if track not in self.question_pools:
                    self.question_pools[track] = {}
                
                for level, questions in levels.items():
                    level_int = int(level)
                    if level_int not in self.question_pools[track]:
                        self.question_pools[track][level_int] = []
                    
                    self.question_pools[track][level_int].extend(questions)
            
            return True
        except Exception as e:
            print(f"Error loading questions: {e}")
            return False


# Compatibility layer for existing adaptive learning system
class QuestionManager:
    """Wrapper class to maintain compatibility with existing code"""
    
    def __init__(self):
        self.generator = AdaptiveMCQGenerator()
        
    def get_questions_for_track(self, track: str) -> Dict:
        """Get all questions for a track in the old format"""
        if track not in self.generator.question_pools:
            return {}
        
        return self.generator.question_pools[track]


# Initialize the global question manager and create QUESTIONS for backward compatibility
_question_manager = QuestionManager()
QUESTIONS = {}

# Populate QUESTIONS dictionary for backward compatibility
for track in _question_manager.generator.get_available_tracks():
    QUESTIONS[track] = _question_manager.get_questions_for_track(track)


# Enhanced functions for the adaptive learning system
def get_adaptive_question(track: str, level: int, used_questions: List[str] = None) -> Optional[Dict]:
    """
    Get an adaptive question for the specified track and level
    
    Args:
        track: Technology track
        level: Difficulty level (1-3)
        used_questions: List of already used question texts
        
    Returns:
        Question dictionary or None
    """
    return _question_manager.generator.get_question(track, level, used_questions or [])


def generate_interview_questions(track: str, student_ability: float = 0.5, count: int = 10) -> List[Dict]:
    """
    Generate a complete set of interview questions
    
    Args:
        track: Technology track
        student_ability: Current student ability level (0.0-1.0)
        count: Number of questions to generate
        
    Returns:
        List of adaptive questions
    """
    return _question_manager.generator.get_adaptive_question_set(track, student_ability, count)


def get_question_statistics(track: str = None) -> Dict:
    """Get statistics about available questions"""
    if track:
        return _question_manager.generator.get_track_statistics(track)
    
    stats = {"all_tracks": {}}
    for track_name in _question_manager.generator.get_available_tracks():
        stats["all_tracks"][track_name] = _question_manager.generator.get_track_statistics(track_name)
    
    return stats


def add_custom_questions(track: str, level: int, questions: List[Dict]):
    """Add custom questions to the pool"""
    _question_manager.generator.update_question_pool(track, level, questions)


# Cache warming function
@st.cache_data
def warm_question_cache():
    """Pre-load questions for better performance"""
    cached_data = {}
    for track in _question_manager.generator.get_available_tracks():
        cached_data[track] = _question_manager.get_questions_for_track(track)
    return cached_data


# Test function to validate the system
def test_question_system():
    """Test the question generation system"""
    print("🧪 Testing Enhanced Question System")
    print("=" * 50)
    
    for track in ["web", "ai", "cyber", "data", "mobile", "devops"]:
        print(f"\n📚 Testing {track.upper()} track:")
        
        # Test getting questions at different levels
        for level in [1, 2, 3]:
            question = get_adaptive_question(track, level)
            if question:
                print(f"  Level {level}: ✅ {question['text'][:50]}...")
            else:
                print(f"  Level {level}: ❌ No questions available")
        
        # Test adaptive question set
        adaptive_set = generate_interview_questions(track, 0.5, 5)
        print(f"  Adaptive set: ✅ Generated {len(adaptive_set)} questions")
        
        # Test statistics
        stats = get_question_statistics(track)
        print(f"  Statistics: ✅ {stats['total_questions']} total questions")


if __name__ == "__main__":
    # Run tests when script is executed directly
    test_question_system()
    
    
    print("\n🚀 Enhanced Question System Ready!")
    print(f"📊 Available tracks: {', '.join(_question_manager.generator.get_available_tracks())}")
    print("💡 Use get_adaptive_question() for single questions")
    print("🎯 Use generate_interview_questions() for complete sets")
    print("📈 Use get_question_statistics() for analytics")