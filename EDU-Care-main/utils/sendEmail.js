const nodemailer = require("nodemailer");

const sendEmail = async (option) => {
  // Input validation
  if (!option.email || !option.subject || !option.message) {
    throw new Error("Missing required fields: email, subject, or message");
  }

  try {
    // 1- Create transporter
    const transporter = nodemailer.createTransport({
      host: process.env.EMAIL_HOST,
      port: process.env.EMAIL_PORT,
      service: "gmail", // If using Gmail as service provider
      auth: {
        user: process.env.EMAIL_USERNAME,
        pass: process.env.EMAIL_PASSWORD,
      },
    });

    // 2- Define mail options
    const mailOpts = {
      from: `"EDU-CARE" <${process.env.EMAIL_USERNAME}>`,
      to: option.email,
      subject: option.subject,
      text: option.message,
    };

    // 3- Send email
    await transporter.sendMail(mailOpts);
    console.log("Email sent successfully");
  } catch (error) {
    console.error("Error sending email:", error);
    throw error; // Re-throw to let the caller handle it
  }
};

module.exports = sendEmail;
