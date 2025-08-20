const getGradingScores = async (filePath) => {
  // Mock response: generate random scores and a simple HTML report
  const similarityScore = parseFloat((Math.random() * 100).toFixed(2));
  const aiGeneratedScore = parseFloat((Math.random() * 100).toFixed(2));
  const plagiarismScore = parseFloat((Math.random() * 100).toFixed(2));
  const reportHtml = `<html><body><h2>Report</h2><p>Similarity: ${similarityScore}%</p><p>AI Score: ${aiGeneratedScore}%</p><p>Plagiarism: ${plagiarismScore}%</p></body></html>`;
  return { similarityScore, aiGeneratedScore, plagiarismScore, reportHtml };
};

module.exports = { getGradingScores };
