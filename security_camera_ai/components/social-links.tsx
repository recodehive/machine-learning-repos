'use client'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faGithub, faLinkedin } from '@fortawesome/free-brands-svg-icons';

const SocialMediaLinks = () => {
    return (
        <div className='flex flex-row gap-4'>
            <a href="https://github.com/VAIBHAV-KESARWANI/SECURITY-CAMERA-AI" target="_blank" rel="noopener noreferrer">
                <FontAwesomeIcon icon={faGithub} size="2x" />
            </a>
            <a href="https://www.linkedin.com/in/vaibhav-kesarwani-9b5b35252/" target="_blank" rel="noopener noreferrer">
                <FontAwesomeIcon icon={faLinkedin} size="2x" />
            </a>
            {/* <a href="https://vaibhavkesarwani.vercel.app/" target="_blank" rel="noopener noreferrer"> */}
                {/* <FontAwesomeIcon icon={faGlobe} className='fa-solid fa-globe' size="2x" /> */}
            {/* </a> */}
        </div>
    );
};

export default SocialMediaLinks;
