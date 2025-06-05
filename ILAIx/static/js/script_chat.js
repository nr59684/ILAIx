document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('circleContainer');
    const numCircles = 100;
  
    for (let i = 0; i < numCircles; i++) {
      const circle = document.createElement('div');
      circle.className = 'circle';
      
      // Random styling
      const size = Math.random() * 8 + 2 + 'px';
      const startY = Math.random() * 100 + 100 + 'vh';
      const duration = (28 + Math.random() * 9) + 's';
      const delay = Math.random() * 37 + 's';
      
      circle.style.width = size;
      circle.style.height = size;
      circle.style.animationDuration = duration;
      circle.style.animationDelay = delay;
      circle.style.animationName = `move-${i}`;
      
      // Create unique keyframe animation
      const style = document.createElement('style');
      style.textContent = `
        @keyframes move-${i} {
          from { transform: translate(${Math.random() * 100}vw, ${startY}); }
          to { transform: translate(${Math.random() * 100}vw, ${-startY}); }
        }
      `;
      document.head.appendChild(style);
      
      container.appendChild(circle);
    }
  });
  
async function postData(url = "", data = {}) { 
    const response = await fetch(url, {
      method: "POST", headers: {
        "Content-Type": "application/json", 
      }, body: JSON.stringify(data),  
    });
    return response.json(); 
  }


sendButton.addEventListener("click", async ()=>{ 
    questionInput = document.getElementById("questionInput").value;
    modelName = document.getElementById("dropdownButton").innerText;
    document.getElementById("questionInput").value = "";
    document.querySelector(".right2").style.display = "block"
    document.getElementById("selectedModel").innerHTML = `<strong>${modelName}</strong>`;
    document.querySelector(".right1").style.display = "none"
    question2.innerHTML = questionInput;
    // Get the answer and populate it! 
    let result = await postData("/api", {"question": questionInput,"modelName": modelName})
    solution.innerHTML = result.answer
})

let explanations = null;
explainButton.addEventListener("click", async () => {
    explainButton.disabled = true;
    explainButton.style.display = "none";
    let newanswerHTML = "";
    let solution = document.getElementById("solution").innerHTML
    let labels = solution.split('<br>');
    let labelIndex = 0; // Keep track of label index
    // Create buttons for each label
    labels.forEach(label => {
        newanswerHTML += `
            <button class="label-button" data-label-index="${labelIndex}">${label}</button><br>
        `;
        labelIndex++;
    });
    const path = window.location.pathname;
    const parts = path.split('/');
    // The last part of the path will be the license name (e.g., 'Xerox')
    const licenseName = parts[parts.length - 1];
    questionInput = document.getElementById("question2").innerHTML;
    let result = await postData("/getExplanations", {"question": questionInput,"licenseName": licenseName})
    explanations=result
    console.log(explanations)
    document.getElementById("solution").innerHTML = newanswerHTML;
    const labelButtons = document.querySelectorAll(".label-button");
    labelButtons.forEach(button => {
        button.addEventListener("click", () => {
            const index = button.dataset.labelIndex;
            // Call a function to highlight the corresponding reason in the input text
            toggleHighlight(index, button);
        });
    });
});

const usedColors = new Set();

// Function to generate a random light color (lightness: 80%)
function getRandomLightColor() {
  let color;
  // Continue generating until a unique color is produced
  do {
    const hue = Math.floor(Math.random() * 360); // Random hue from 0 to 359
    color = `hsl(${hue}, 70%, 50%)`; // 70% saturation and 80% lightness for a light tone
  } while (usedColors.has(color));
  usedColors.add(color);
  return color;
}

document.getElementById("question2").addEventListener("click", function(event) {
  // Check if the clicked element is a highlighted span
  if (event.target.classList.contains("highlighted")) {
    const labelIndex = event.target.dataset.labelIndex;
    // Find the corresponding button using the same data-label-index
    const correspondingButton = document.querySelector(`.label-button[data-label-index="${labelIndex}"]`);
    if (correspondingButton) {
      correspondingButton.scrollIntoView({ behavior: "smooth", block: "center" });
      
      // Optionally, add a temporary highlight to the button for visual feedback
      correspondingButton.classList.add("highlighted-button");
      setTimeout(() => {
         correspondingButton.classList.remove("highlighted-button");
      }, 2000);
    }
  }
});


function toggleHighlight(index, button) {
  // Get the original license text from the input field
  const questionElement = document.getElementById("question2");
  let inputText = questionElement.innerHTML;
  
  // Get the reason text corresponding to the clicked label
  const labels = Object.keys(explanations);
  const reason = explanations[labels[index]];
  
  // Use previously stored color if available, otherwise generate a new one
  let assignedColor = button.dataset.color;
  if (!assignedColor) {
    assignedColor = getRandomLightColor();  // Assuming this function is defined as before
    button.dataset.color = assignedColor;
  }
  
  // Toggle the active state of the button
  const isHighlighted = button.classList.toggle("active");

  if (isHighlighted) {
    button.style.backgroundColor = assignedColor;
    button.style.color = "white"; // Ensure contrast
  } else {
    button.style.backgroundColor = "lightslategrey"; // Reset background color
    button.style.color = "white";
  }
  
  // Update the input text by highlighting/unhighlighting phrases.
  reason.forEach(phrase => {
    const escapedPart = phrase.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&'); // Escape regex special chars
    if (isHighlighted) {
      // Add a data attribute to know which label (index) generated this highlight.
      inputText = inputText.replace(
        new RegExp(`(${escapedPart})`, 'gi'),
        `<span class="highlighted" data-label-index="${index}" style="color: ${assignedColor}; font-weight: bold;">$1</span>`
      );
    } else {
      inputText = inputText.replace(
        new RegExp(`(${escapedPart})`, 'gi'),
        `<span style="color: white; font-weight: normal;">$1</span>`
      );
    }
  });
  
  questionElement.innerHTML = inputText;
  
  // If highlighted, scroll to the first highlighted text
  if (isHighlighted) {
    setTimeout(() => {
      const firstHighlight = document.querySelector("#question2 .highlighted");
      if (firstHighlight) {
        firstHighlight.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    }, 100);
  }
}

